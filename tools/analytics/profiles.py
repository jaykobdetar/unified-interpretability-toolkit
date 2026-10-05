"""Locally reviewed, exact-digest layout profiles. No model execution/imports."""

import importlib.util
from pathlib import Path

from .source import read_layout_evidence

LAYOUT = "separate-contiguous-linear-out-in-v1"
LINEAR_SHA = "fa22acbb48e41bff3be26779c2f242b15663b8157e54f0d55987721add97d16f"
PROFILES = {
    "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843": {
        "model_type": "llama",
        "implementation": "31bf660a663259134324bc65da4e155951dc89c5ca46471d2325a9938e859e26",
        "model": "HuggingFaceTB/SmolLM2-135M",
        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    },
    "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30": {
        "model_type": "qwen3",
        "implementation": "4b95c371fd26d40c69083dab36ac1eafd8cf82b415a0bb827275097c5ad2305b",
        "model": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    },
}


def resolve_local(model_root):
    """Return analyze() options only if exact reviewed local evidence still matches."""
    try:
        # read_layout_evidence checks file types, bounds and before/after identity.
        # First choose the path from a bounded config read, then validate the exact
        # same config digest again with the official implementation source.
        config_path = Path(model_root) / "config.json"
        config, probe = read_layout_evidence(config_path, config_path)
        profile = PROFILES.get(probe["config_sha256"])
        if not profile or config.get("model_type") != profile["model_type"]:
            return {}, "No reviewed exact configuration profile"
        transformers = importlib.util.find_spec("transformers")
        torch = importlib.util.find_spec("torch")
        if not transformers or not torch:
            return {}, "Reviewed installed implementation source unavailable"
        implementation = (
            Path(transformers.origin).parent
            / "models"
            / profile["model_type"]
            / f"modeling_{profile['model_type']}.py"
        )
        config, evidence = read_layout_evidence(config_path, implementation)
        if (
            evidence["config_sha256"] != probe["config_sha256"]
            or evidence["implementation_sha256"] != profile["implementation"]
        ):
            return {}, "Configuration or official implementation digest changed"
        linear = Path(torch.origin).parent / "nn/modules/linear.py"
        _, linear_evidence = read_layout_evidence(config_path, linear)
        if (
            linear_evidence["implementation_sha256"] != LINEAR_SHA
            or linear_evidence["config_sha256"] != probe["config_sha256"]
        ):
            return {}, "PyTorch Linear layout evidence changed"
        evidence.update(
            model=profile["model"],
            revision=profile["revision"],
            linear_implementation_sha256=LINEAR_SHA,
            implementation_package="transformers 4.56.2; source files only, no model imported",
        )
        return {
            "config": config,
            "evidence": evidence,
            "reviewed_profiles": {
                (evidence["config_sha256"], evidence["implementation_sha256"]): LAYOUT
            },
        }, None
    except (OSError, ValueError, TypeError, AttributeError, ImportError):
        return {}, "Missing or invalid local layout evidence; head labels disabled"
