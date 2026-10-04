"""No ML imports: architecture, independent head math, full-layer planning and budgets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from inference_architecture import ARCH, CONFIG, MANIFEST, describe, shapes
import inference_sweep as sweep
from inference_edits import SOURCE_MODEL
from sweep_contracts import Parameter, Torch, Vec, apply, admitted, request

class Contracts(unittest.TestCase):
    def full(self):
        data=request();data['targets']=[{'kind':'layer_heads','layer':2}]
        return data,sweep.build_plan(data,False)

    def test_verified_config_and_nondefault_gqa_dimensions(self):
        self.assertEqual(ARCH['query_heads']//ARCH['kv_heads'],3)
        self.assertEqual([h//ARCH['queries_per_kv'] for h in range(9)],[0,0,0,1,1,1,2,2,2])
        config={**CONFIG,'hidden_size':24,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':8,'num_hidden_layers':2,'intermediate_size':40,'vocab_size':60}
        arch=describe(config);mapping=shapes(arch)
        self.assertEqual(mapping['model.layers.1.self_attn.o_proj.weight'],[24,32])
        self.assertEqual(mapping['model.layers.1.self_attn.k_proj.weight'],[16,24])
        self.assertEqual(len(mapping),15)
        self.assertEqual(arch['queries_per_kv'],2)
        for change in [{'model_type':'qwen3'},{'architectures':['CustomCode']},{'num_attention_heads':True},{'num_key_value_heads':2},{'attention_bias':True},{'tie_word_embeddings':False},{'head_dim':False}]:
            with self.subTest(change=change),self.assertRaises(ValueError):describe({**CONFIG,**change})
        for change in [{'hidden_size':577},{'num_key_value_heads':0}]:
            with self.assertRaises(ValueError):describe({**CONFIG,**change})
        raw=(Path(__file__).resolve().parents[1]/'docs/models/smollm2-135m-config.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),MANIFEST['files']['config.json'])

    def test_adapter_rejects_unreviewed_runtime_layout(self):
        from types import SimpleNamespace
        from inference_architecture import verify_attention_layout
        class Model: pass
        class Block: pass
        class Attention: pass
        class Linear:
            def __init__(self,out_features,in_features):
                self.out_features,self.in_features,self.bias=out_features,in_features,None
        model=Model();model.training=False;model.config=SimpleNamespace(_attn_implementation='eager')
        blocks=[]
        for _ in range(ARCH['layers']):
            block=Block();attn=Attention();block.self_attn=attn
            attn.head_dim=64;attn.num_key_value_groups=3
            for name,out_features,in_features in [('q_proj',576,576),('k_proj',192,576),('v_proj',192,576),('o_proj',576,576)]:
                setattr(attn,name,Linear(out_features,in_features))
            blocks.append(block)
        model.model=SimpleNamespace(layers=blocks)
        modules={'torch':SimpleNamespace(nn=SimpleNamespace(Linear=Linear)),
                 'transformers':SimpleNamespace(__version__='4.56.2'),
                 'transformers.models.llama.modeling_llama':SimpleNamespace(LlamaForCausalLM=Model,LlamaAttention=Attention,LlamaDecoderLayer=Block)}
        with patch.dict(sys.modules,modules):
            verify_attention_layout(model)
            for obj,key,value in [(model,'training',True),(model.config,'_attn_implementation','sdpa'),
                                  (blocks[0].self_attn,'num_key_value_groups',1),(blocks[0].self_attn.o_proj,'in_features',192),
                                  (modules['transformers'],'__version__','unreviewed')]:
                prior=getattr(obj,key);setattr(obj,key,value)
                with self.assertRaises(ValueError):verify_attention_layout(model)
                setattr(obj,key,prior)

    def test_output_columns_and_query_rows_are_separate(self):
        head={'kind':'head','layer':0,'head':2}
        output=sweep._edits(head,'zero',None)[0]
        query=sweep._edits({**head,'kind':'query_head'},'zero',None)[0]
        self.assertEqual((output['kind'],output['start'],output['end']),('columns',128,192))
        self.assertTrue(output['tensor'].endswith('.o_proj.weight'))
        self.assertEqual(query['kind'],'rows');self.assertTrue(query['tensor'].endswith('.q_proj.weight'))
        self.assertEqual(len(sweep.selected_indices([output])[output['tensor']]),576*64)
        # Independent 2-head output projection: removing a column block equals
        # removing that head's post-attention contribution. No tensor library.
        matrix=[[1.,2.,3.,4.],[-2.,1.,4.,3.]];z=[2.,-1.,3.,2.]
        baseline=[sum(w*x for w,x in zip(row,z)) for row in matrix]
        ablated=[sum(row[i]*z[i] for i in (2,3)) for row in matrix]
        contribution=[sum(row[i]*z[i] for i in (0,1)) for row in matrix]
        self.assertEqual([b-a for b,a in zip(baseline,ablated)],contribution)
        # A zero query gives uniform probabilities over allowed keys, generally
        # a nonzero value average, not a removed output head.
        values=[[2.,4.],[6.,8.]]
        self.assertEqual([sum(v[i] for v in values)/2 for i in range(2)],[4.,6.])

    def test_all_heads_one_plan_comparison_controls_and_cap(self):
        data,plan=self.full()
        self.assertEqual((len(plan['targets']),plan['records'],plan['prefills']),(9,19,38))
        self.assertEqual([t['head'] for t in plan['targets']],list(range(9)))
        self.assertEqual(plan['scope'],'layer_heads')
        self.assertIn('not a null/no-effect',plan['control_semantics'])
        for i in range(9):
            target,control=plan['cases'][1+2*i:3+2*i]
            self.assertEqual(target['selected_cells'],control['selected_cells'])
            self.assertNotEqual(target['target']['head'],control['target']['head'])
            self.assertIn(control['target'],plan['targets'])
            for case in (target,control):
                self.assertEqual(case['edits'][0]['kind'],'columns')
                self.assertEqual(case['edits'][0]['operation'],'zero')
                self.assertTrue(case['edits'][0]['tensor'].endswith('o_proj.weight'))
        self.assertEqual(plan['limits']['wall_seconds'],120);self.assertEqual(plan['limits']['worker_cpu_seconds'],90)
        for change in [{'prompts':['a','b']},{'operation':'scale','scale':0},{'targets':[{'kind':'layer_heads','layer':30}]},{'targets':[{'kind':'layer_heads','layer':2},{'kind':'head','layer':2,'head':0}]}]:
            with self.assertRaises(ValueError):sweep.build_plan({**data,**change},False)
        accepted={**data,'plan_digest':plan['digest']};self.assertEqual(sweep.build_plan(accepted),plan)
        with self.assertRaises(ValueError):sweep.build_plan({**accepted,'seed':18})

    def test_partial_coverage_requires_target_and_control(self):
        _,plan=self.full()
        partial=sweep.coverage(plan,2)
        self.assertEqual(partial['unfinished_heads'],list(range(9)))
        partial=sweep.coverage(plan,3)
        self.assertEqual(partial['unfinished_heads'],list(range(1,9)))
        self.assertEqual(partial['finished_targets'],[{'kind':'head','layer':2,'head':0}])
        self.assertFalse(partial['complete'])
        done=sweep.coverage(plan,19);self.assertEqual(done['unfinished_heads'],[]);self.assertTrue(done['complete'])

    def test_full_plan_runs_one_sequence_restoring_between_all_cases(self):
        from contextlib import contextmanager
        from types import SimpleNamespace
        data,plan=self.full();events=[];state={'edited':False};calls=[]
        @contextmanager
        def temporary(_torch,_model,edits,_source):
            self.assertFalse(state['edited']);state['edited']=bool(edits)
            try:
                cells=0 if not edits else 576*64
                yield {'selected_cells':cells,'changed_cells':cells,'parameter_delta_l2':float(bool(edits))}
            finally:state['edited']=False
        def generate(_torch,_tokenizer,_model,prompt,limit,layer,record,scores,activation_site):
            calls.append(state['edited']);value=0.5 if state['edited'] else 1.
            scores(Vec([value,0.]));record({'type':'step','activation':[value]*576,'layer':layer,
                'activation_site':activation_site,'activation_kind':'fixture','position':0,'input_token_id':1,
                'compute_ms':1.,'top_logits':[{'id':0,'value':value}]})
        with patch.object(sweep,'temporary_edits',side_effect=temporary):
            sweep.run(Torch,SimpleNamespace(decode=lambda *args,**kw:'fixture'),None,data,plan,120,generate,events.append,clock=lambda:0)
        self.assertEqual(len(calls),38);self.assertTrue(all(not x for x in calls[::2]))
        self.assertEqual(len([e for e in events if e['type']=='step']),19)
        self.assertTrue(events[-1]['coverage']['complete']);self.assertFalse(state['edited'])

    def test_cpu_budget_stops_before_any_forward_without_new_allowance(self):
        data,plan=admitted();events=[]
        def forbidden(*args,**kwargs):raise AssertionError('No forward after CPU margin')
        sweep.run(Torch,None,None,data,plan,120,forbidden,events.append,clock=lambda:0,cpu_deadline=90,cpu_clock=lambda:86)
        self.assertEqual(events[-1]['status'],'time_limit');self.assertEqual(events[-1]['coverage']['completed_ids'],[])

    def test_budget_after_baseline_leaves_case_unfinished_and_model_pristine(self):
        data,plan=admitted();events=[];calls=[]
        name='model.layers.0.self_attn.q_proj.weight';model={name:Parameter()}
        def generate(*args,**kwargs):
            calls.append(1);args[6]({'type':'step','top_logits':[{'id':0,'value':1.}]});args[7](Vec([1.,0.]))
        clock=iter([0.,116.])
        with patch.object(sweep,'temporary_edits',side_effect=AssertionError('No edit after baseline exhausted budget')):
            sweep.run(Torch,None,model,data,plan,120,generate,events.append,clock=lambda:next(clock))
        self.assertEqual(len(calls),1);self.assertEqual(events[-1]['coverage']['completed_ids'],[])
        self.assertEqual(events[-1]['status'],'time_limit')

if __name__=='__main__':unittest.main()
