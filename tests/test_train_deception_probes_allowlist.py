from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "training" / "train_deception_probes.py"

spec = importlib.util.spec_from_file_location("train_deception_probes", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_load_and_filter_allowlist(tmp_path: Path) -> None:
    allowlist_path = tmp_path / "allowlist.json"
    payload = {"ids": ["a", "c"]}
    allowlist_path.write_text(json.dumps(payload), encoding="utf-8")

    allowlist = module.load_id_allowlist(str(allowlist_path))
    items = [
        {"id": "a", "label": 0},
        {"id": "b", "label": 1},
        {"id": "c", "label": 0},
    ]
    filtered = module.filter_items_by_allowlist(items, allowlist)

    assert allowlist == {"a", "c"}
    assert [item["id"] for item in filtered] == ["a", "c"]
