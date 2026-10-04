"""Pinned configuration and explicitly supported built-in attention layout; no ML imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'docs/models/smollm2-135m.json').read_text())
CAPTURE_SITES = {
    'block': 'decoder block output after attention and MLP residual additions, before final RMSNorm',
    'attention': 'attention output after o_proj, before attention residual addition',
    'mlp': 'MLP output after down_proj, before MLP residual addition',
}


def describe(config):
    if config.get('model_type') != 'llama' or config.get('architectures') != ['LlamaForCausalLM']:
        raise ValueError('No verified attention layout for this architecture')
    keys = ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'vocab_size')
    if any(type(config.get(k)) is not int or config[k] <= 0 for k in keys):
        raise ValueError('Invalid architecture dimensions')
    width, heads, kv = (config[k] for k in ('hidden_size','num_attention_heads','num_key_value_heads'))
    dim = config.get('head_dim')
    if dim is None:
        if width % heads: raise ValueError('Head dimension cannot be inferred exactly')
        dim = width // heads
    if type(dim) is not int or dim <= 0 or heads % kv:
        raise ValueError('Invalid GQA grouping or head dimension')
    if config.get('attention_bias', False) or config.get('mlp_bias', False) or config.get('tie_word_embeddings') is not True:
        raise ValueError('Only the verified bias-free, tied Llama layout is supported')
    return {'model_type':'llama', 'width':width, 'layers':config['num_hidden_layers'],
            'query_heads':heads, 'kv_heads':kv, 'head_dim':dim, 'queries_per_kv':heads//kv,
            'vocab_size':config['vocab_size'], 'intermediate_size':config['intermediate_size'],
            'capture_sites':dict(CAPTURE_SITES), 'layout':'llama-eager-head-major-v1',
            'head_ablation':'o_proj columns', 'query_intervention':'q_proj rows'}


raw = (ROOT / 'docs/models/smollm2-135m-config.json').read_bytes()
if hashlib.sha256(raw).hexdigest() != MANIFEST['files']['config.json']:
    raise ValueError('Bundled configuration differs from the pinned disk config')
CONFIG = json.loads(raw)
ARCH = describe(CONFIG)
WIDTH, LAYERS, HEADS, KV_HEADS, HEAD_DIM, VOCAB = (ARCH[k] for k in ('width','layers','query_heads','kv_heads','head_dim','vocab_size'))


def shapes(arch=ARCH):
    w, d, h, kv, inner = (arch[k] for k in ('width','head_dim','query_heads','kv_heads','intermediate_size'))
    result = {'model.embed_tokens.weight':[arch['vocab_size'],w]}
    for layer in range(arch['layers']):
        for projection, shape in {'self_attn.q_proj':[h*d,w], 'self_attn.k_proj':[kv*d,w],
                'self_attn.v_proj':[kv*d,w], 'self_attn.o_proj':[w,h*d],
                'mlp.gate_proj':[inner,w], 'mlp.up_proj':[inner,w], 'mlp.down_proj':[w,inner]}.items():
            result[f'model.layers.{layer}.{projection}.weight'] = shape
    return result


def head_layout_descriptor():
    """Public configuration evidence only; never an execution or fit receipt."""
    return {'schema':'weight-atlas-head-layout-v1', 'adapter_id':'builtin-llama-eager', 'adapter_version':1,
            'source_model':{'repo':MANIFEST['repo'], 'revision':MANIFEST['revision'],
                            'weights_sha256':MANIFEST['files']['model.safetensors'],
                            'config_sha256':MANIFEST['files']['config.json']},
            **ARCH, 'evidence':'pinned_configuration', 'runtime_verified':False,
            'inference_support':'requires complete pinned files, supported runtime and resource admission',
            'native_weight_layout':['output_feature','input_feature'],
            'projection_mappings':{
                'q_proj':{'axis':'rows','head_kind':'query','heads':HEADS,'head_dim':HEAD_DIM,'shape':[HEADS*HEAD_DIM,WIDTH]},
                'k_proj':{'axis':'rows','head_kind':'kv','heads':KV_HEADS,'head_dim':HEAD_DIM,'shape':[KV_HEADS*HEAD_DIM,WIDTH]},
                'v_proj':{'axis':'rows','head_kind':'kv','heads':KV_HEADS,'head_dim':HEAD_DIM,'shape':[KV_HEADS*HEAD_DIM,WIDTH]},
                'o_proj':{'axis':'columns','head_kind':'query_output','heads':HEADS,'head_dim':HEAD_DIM,'shape':[WIDTH,HEADS*HEAD_DIM]}},
            'ranges':'zero-based half-open: [head * head_dim, (head + 1) * head_dim)',
            'query_to_kv':'floor(query_head / queries_per_kv)',
            'kv_to_queries':'[kv_head * queries_per_kv, (kv_head + 1) * queries_per_kv)',
            'slice_support':'stored rank-two matrices only; canonical empty slice; no higher-rank inference handoff'}


def verify_attention_layout(model):
    # This adapter is source-reviewed for this installed implementation only.
    # New versions/architectures require a new review, never a guessed layout.
    import transformers
    import torch
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, LlamaDecoderLayer
    if transformers.__version__ != '4.56.2' or type(model) is not LlamaForCausalLM or model.training:
        raise ValueError('Unverified built-in Llama implementation or training mode')
    if model.config._attn_implementation != 'eager' or len(model.model.layers) != LAYERS:
        raise ValueError('Unverified attention implementation or layer count')
    for block in model.model.layers:
        attn = block.self_attn
        if (type(block) is not LlamaDecoderLayer or type(attn) is not LlamaAttention or
                attn.head_dim != HEAD_DIM or attn.num_key_value_groups != HEADS//KV_HEADS):
            raise ValueError('Attention head layout differs from verified GQA mapping')
        for key, out_features, in_features in [('q_proj',HEADS*HEAD_DIM,WIDTH),
                ('k_proj',KV_HEADS*HEAD_DIM,WIDTH), ('v_proj',KV_HEADS*HEAD_DIM,WIDTH),
                ('o_proj',WIDTH,HEADS*HEAD_DIM)]:
            linear = getattr(attn,key)
            if (type(linear) is not torch.nn.Linear or linear.bias is not None or
                    linear.out_features != out_features or linear.in_features != in_features):
                raise ValueError('Projection is not the verified output-by-input linear layout')
    return {**head_layout_descriptor(), 'evidence':'loaded_builtin_layout', 'runtime_verified':True,
            'runtime':{'transformers':transformers.__version__, 'attention_backend':'eager', 'device':'cpu', 'dtype':'float32'}}


def bind_viewer_head_layout(model_info, trusted_binding=None):
    """Project host-validated receipt correspondence; no validation I/O or admission.

    trusted_binding is an internal host result, NEVER an HTTP/visitor field.
    The host must have checked the installed receipt's current fingerprints and
    correspondence to this renderer before constructing its four-field binding.
    This adapter checks correspondence, not the authenticity of an arbitrary dict.
    It never infers a binding from a path, revision label or copied descriptor.
    """
    result = dict(model_info)
    # Never forward an upstream/stale/runtime-owned annotation without rebinding.
    result.pop('head_layout', None)
    result.pop('head_layout_binding', None)
    fields = {'source_identity','model_identity','weights_sha256','config_sha256'}
    if (type(trusted_binding) is not dict or set(trusted_binding)!=fields or
            any(type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                for v in trusted_binding.values())):
        return result
    descriptor = head_layout_descriptor()
    source = descriptor['source_model']
    if (trusted_binding['source_identity'] != model_info.get('source_identity') or
            trusted_binding['model_identity'] != model_info.get('model_identity') or
            trusted_binding['weights_sha256'] != source['weights_sha256'] or
            trusted_binding['config_sha256'] != source['config_sha256'] or
            model_info.get('revision') != source['revision'] or
            'comparison_identity' in model_info or 'coordinate_space' in model_info):
        return result
    result['head_layout'] = descriptor
    result['head_layout_binding'] = dict(trusted_binding)
    return result
