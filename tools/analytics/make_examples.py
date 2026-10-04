"""Synthetic fixtures only; no source/model reads or SVD calls."""
from pathlib import Path
import json
from .core import analyze, digest


def examples():
    config = {'hidden_size': 4, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'num_hidden_layers': 1}
    # This profile is deliberately fixture-only. It is not a real architecture attestation.
    evidence = {'config_sha256': 'SYNTHETIC-FIXTURE', 'config_canonical_sha256': digest(config),
                'implementation_sha256': 'SYNTHETIC-CONTIGUOUS-LAYOUT'}
    return {
        'matrix': analyze(list(range(1, 17)), source_identity='SYNTHETIC-NOT-MODEL-EVIDENCE',
                          tensor='model.layers.0.self_attn.q_proj.weight', shape=[4, 4],
                          region={'row': 0, 'col': 0, 'rows': 4, 'cols': 4}, seed=42,
                          config=config, evidence=evidence,
                          reviewed_profiles={('SYNTHETIC-FIXTURE', 'SYNTHETIC-CONTIGUOUS-LAYOUT'): 'separate-contiguous-linear-out-in-v1'}),
        'vector': analyze([.2, -.9, .1, -.5, 0., .6, -.3, .7], source_identity='SYNTHETIC-NOT-MODEL-EVIDENCE',
                          tensor='synthetic.vector', shape=[8], region={'row': 0, 'col': 0, 'rows': 1, 'cols': 8}, seed=42)
    }


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[2]/'web/analytics-examples.json'
    path.write_text(json.dumps(examples(), indent=2, allow_nan=False)+'\n')
