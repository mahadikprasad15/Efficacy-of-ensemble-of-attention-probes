from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "actprobe" / "src"))

from actprobe.probes.ensemble import ProbeLogitGatedEnsemble, mean_entropy  # noqa: E402


def load_pipeline_module():
    module_path = REPO_ROOT / "scripts" / "pipelines" / "run_probe_gated_ensemble_matrix.py"
    spec = importlib.util.spec_from_file_location("run_probe_gated_ensemble_matrix", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_logit_gated_ensemble_outputs_valid_weights() -> None:
    model = ProbeLogitGatedEnsemble(num_experts=4, hidden_dim=8, dropout=0.0, temperature=1.0)
    logits = torch.tensor([[1.0, -1.0, 0.5, 2.0], [0.0, 0.5, -0.5, 1.0]], dtype=torch.float32)
    combined, weights = model(logits, return_weights=True)

    assert combined.shape == (2, 1)
    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(weights >= 0).item()


def test_mean_entropy_matches_uniform_vs_peaked_expectation() -> None:
    uniform = torch.full((3, 4), 0.25, dtype=torch.float32)
    peaked = torch.tensor(
        [
            [0.97, 0.01, 0.01, 0.01],
            [0.90, 0.05, 0.03, 0.02],
            [0.88, 0.04, 0.04, 0.04],
        ],
        dtype=torch.float32,
    )
    assert mean_entropy(uniform).item() > mean_entropy(peaked).item()


def test_temperature_controls_weight_sharpness() -> None:
    logits = torch.tensor([[3.0, 1.0, -1.0]], dtype=torch.float32)
    low_tau = ProbeLogitGatedEnsemble(num_experts=3, hidden_dim=4, dropout=0.0, temperature=0.5)
    high_tau = ProbeLogitGatedEnsemble(num_experts=3, hidden_dim=4, dropout=0.0, temperature=2.0)

    with torch.no_grad():
        low_tau.gate_net[0].weight.zero_()
        low_tau.gate_net[0].bias.zero_()
        low_tau.gate_net[3].weight.zero_()
        low_tau.gate_net[3].bias.copy_(torch.tensor([3.0, 1.0, -1.0]))
        high_tau.load_state_dict(low_tau.state_dict())

    low_weights = low_tau.gate_weights(logits)
    high_weights = high_tau.gate_weights(logits)
    assert float(low_weights.max().item()) > float(high_weights.max().item())


def test_stratified_gate_split_keeps_both_classes() -> None:
    module = load_pipeline_module()
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
    train_idx, val_idx = module.stratified_split_indices(labels, train_fraction=0.8, seed=7)
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    assert len(train_idx) == 16
    assert len(val_idx) == 4
    assert set(train_labels.tolist()) == {0, 1}
    assert set(val_labels.tolist()) == {0, 1}


def test_choose_validation_selected_expert_prefers_best_val_auc() -> None:
    module = load_pipeline_module()
    experts = [
        module.ExpertSpec(expert_idx=0, source_dataset="s", pooling="mean", layer=14, probe_path=Path("a.pt")),
        module.ExpertSpec(expert_idx=1, source_dataset="s", pooling="max", layer=15, probe_path=Path("b.pt")),
        module.ExpertSpec(expert_idx=2, source_dataset="s", pooling="attn", layer=15, probe_path=Path("c.pt")),
    ]
    lookup = {
        ("mean", 14): {"val_auc": 0.61, "val_acc": 0.58},
        ("max", 15): {"val_auc": 0.73, "val_acc": 0.60},
        ("attn", 15): {"val_auc": 0.69, "val_acc": 0.64},
    }
    chosen = module.choose_validation_selected_expert(experts, lookup)
    assert chosen.expert_idx == 1


def test_topk_static_plan_selects_a_valid_subset() -> None:
    module = load_pipeline_module()
    train_logits = np.asarray(
        [
            [2.0, 0.5, 0.1, -0.2],
            [1.8, 0.4, -0.2, 0.1],
            [-1.7, -0.2, 0.3, 0.0],
            [-2.1, -0.4, 0.2, -0.1],
        ],
        dtype=np.float32,
    )
    train_labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    val_logits = train_logits.copy()
    val_labels = train_labels.copy()
    plan = module.topk_static_plan(train_logits, train_labels, val_logits, val_labels, k_values=[1, 2, 4])

    assert plan["k"] in {1, 2, 4}
    assert len(plan["selected_indices"]) == plan["k"]
    assert all(0 <= idx < train_logits.shape[1] for idx in plan["selected_indices"])
