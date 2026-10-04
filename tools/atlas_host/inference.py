"""Configuration-evidence correspondence only; no worker request adapter."""
from copy import deepcopy
import re

from .cache import binding
from .common import digest, fields, integer, require
from .registry import content_digest, validate_manifest


def configuration_evidence(manifest, descriptor, source_binding):
    """Bind task16's trusted descriptor to a host receipt without granting execution.

    The descriptor must come from the built-in adapter, never visitor input.
    Loaded runtime evidence belongs to its job/session and is rejected here.
    This returns no worker request or legacy edit-source identity: a separately
    reviewed adapter must construct the existing closed worker request.
    """
    manifest = validate_manifest(manifest)
    require(type(descriptor) is dict
            and descriptor.get('schema') == 'weight-atlas-head-layout-v1'
            and descriptor.get('adapter_id') == 'builtin-llama-eager'
            and type(descriptor.get('adapter_version')) is int
            and descriptor['adapter_version'] == 1, 'Unsupported built-in descriptor')
    require(descriptor.get('evidence') == 'pinned_configuration'
            and descriptor.get('runtime_verified') is False,
            'Runtime evidence is session-owned, not registry metadata')
    selected = binding(source_binding)
    require(len(selected['shape']) == 2 and selected['slice']['leading_indices'] == [],
            'Only original rank-two empty-slice weights can enter a future inference adapter')
    source = descriptor.get('source_model')
    fields(source, ('repo', 'revision', 'weights_sha256', 'config_sha256'))
    digest(source['weights_sha256'])
    digest(source['config_sha256'])
    files = {item['name']: item['sha256'] for item in manifest['files']}
    require(source['repo'] == manifest['repository']
            and source['revision'] == manifest['revision']
            and files.get('model.safetensors') == source['weights_sha256']
            and files.get('config.json') == source['config_sha256'],
            'Descriptor and installed receipt do not match')
    return {'content_digest': content_digest(manifest),
            'adapter_id': descriptor['adapter_id'], 'adapter_version': 1,
            'evidence': 'pinned_configuration', 'source_correspondence': True,
            'runtime_verified': False, 'fit_verified': False, 'inference_ready': False,
            'reason': 'host_worker_adapter_not_active'}


def head_layout_metadata(manifest, descriptor, source_binding):
    """Trusted same-source projection for task15 labels, never a worker envelope.

    Caller must supply renderer-validated binding for this installed receipt. The
    descriptor originates from task16's pinned built-in adapter. Do not expose
    fields on a source/shape mismatch or fall back to guessed head dimensions.
    """
    evidence = configuration_evidence(manifest, descriptor, source_binding)
    require(descriptor.get('native_weight_layout') == ['output_feature', 'input_feature'],
            'Unverified native weight layout')
    match = re.fullmatch(r'model\.layers\.(0|[1-9][0-9]*)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight',
                         source_binding['name'])
    require(match is not None, 'Tensor is not a declared attention projection')
    layer = int(match.group(1))
    integer(descriptor.get('layers'), 1, 200000)
    require(layer < descriptor['layers'], 'Layer outside configured descriptor')
    query = integer(descriptor.get('query_heads'), 1)
    kv = integer(descriptor.get('kv_heads'), 1)
    dim = integer(descriptor.get('head_dim'), 1)
    width = integer(descriptor.get('width'), 1)
    integer(descriptor.get('queries_per_kv'), 1)
    require(query % kv == 0 and descriptor['queries_per_kv'] == query//kv,
            'Invalid GQA descriptor')
    projection = match.group(2)
    mapping = descriptor.get('projection_mappings', {}).get(projection)
    expected_shape = ([width, query*dim] if projection == 'o_proj'
                      else [dim*(query if projection == 'q_proj' else kv), width])
    require(type(mapping) is dict and mapping.get('shape') == expected_shape
            and source_binding['shape'] == expected_shape
            and mapping.get('axis') == ('columns' if projection == 'o_proj' else 'rows')
            and type(mapping.get('heads')) is int
            and mapping['heads'] == (query if projection in ('q_proj', 'o_proj') else kv)
            and type(mapping.get('head_dim')) is int and mapping['head_dim'] == dim,
            'Native tensor does not match declared projection')
    source = descriptor['source_model']
    return {'head_layout': deepcopy(descriptor),
            'head_layout_binding': {'source_identity': source_binding['source_identity'],
                                    'model_identity': source_binding['model_identity'],
                                    'weights_sha256': source['weights_sha256'],
                                    'config_sha256': source['config_sha256']},
            'host_inference_evidence': evidence}
