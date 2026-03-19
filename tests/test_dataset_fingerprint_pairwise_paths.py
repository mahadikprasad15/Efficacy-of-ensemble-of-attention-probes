import importlib.util
import sys
from pathlib import Path


def load_module():
    module_path = Path("scripts/pipelines/run_dataset_fingerprint_pairwise.py")
    spec = importlib.util.spec_from_file_location("run_dataset_fingerprint_pairwise", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_pairwise_pipeline_mirror_root():
    module = load_module()
    path = module.default_pairwise_pipeline_mirror_root("meta-llama_Llama-3.2-3B-Instruct")
    assert path == Path("results") / "ood_evaluation" / "meta-llama_Llama-3.2-3B-Instruct" / "All_dataset_pairwise_results"


def test_default_wrapper_mirror_root():
    module = load_module()
    path = module.default_wrapper_mirror_root("meta-llama_Llama-3.2-3B-Instruct")
    assert path == Path("results") / "ood_evaluation" / "meta-llama_Llama-3.2-3B-Instruct" / "dataset_fingerprint_pairwise"
