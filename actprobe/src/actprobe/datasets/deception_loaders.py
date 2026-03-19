"""
Deception dataset loaders for Apollo Research datasets.

These datasets are for detecting strategic deception (lying) rather than
factual hallucination. Labels are assigned via LLM judge after generation.
"""

import os
import yaml
import json
import random
import ast
from typing import List, Dict, Any, Optional
from .base import BaseDataset
from .templates import DECEPTION_ROLEPLAYING

class DeceptionRoleplayingDataset(BaseDataset):
    """
    Apollo Research Roleplaying Dataset

    Dataset Structure (from YAML):
        - scenario: Context describing a deceptive situation
        - question: User's question to the role-player
        - answer_prefix: Speaker label (e.g., "Student:", "Driver:")
        - honest_completion: Reference truthful response (NOT USED for generation)
        - deceptive_completion: Reference deceptive response (NOT USED for generation)

    Pipeline:
        1. Create prompts from scenarios (neutral, no deception trigger)
        2. Generate completions using target model (e.g., Llama-3.2-3B)
        3. Label completions using LLM judge (Cerebras Llama-8B)
        4. Extract activations from generated tokens
        5. Train probes to detect deception

    Output Format:
        {
            "prompt": str,           # Formatted scenario + question + answer_prefix
            "gold_label": None,      # Will be assigned after generation + labeling
            "metadata": {
                "dataset": "Deception-Roleplaying",
                "id": str,
                "split": str,
                "scenario": str,     # Full scenario text
                "question": str,
                "honest_reference": str,    # For validation only
                "deceptive_reference": str  # For validation only
            }
        }

    Note: Unlike QA datasets, we don't have gold_answers. Instead, we have
    gold_label which is assigned after generation via LLM judge.
    """

    def __init__(self,
                 split: str = "train",
                 limit: Optional[int] = None,
                 data_file: str = "data/apollo_raw/roleplaying/dataset.yaml",
                 random_seed: int = 42):
        """
        Args:
            split: "train", "validation", or "test"
            limit: Maximum number of examples to load
            data_file: Path to downloaded YAML file
            random_seed: Seed for reproducible train/val/test splits
        """
        super().__init__(split=split, limit=limit)
        self.data_file = data_file
        self.random_seed = random_seed

        # Check if data file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Please run: python scripts/download_apollo_data.py"
            )

    def load_data(self):
        """
        Load YAML and create prompts.

        Creates train/val/test splits:
        - 60% train
        - 20% validation
        - 20% test

        Splits are deterministic (based on random_seed).
        """
        # Read YAML
        with open(self.data_file, 'r') as f:
            raw_data = yaml.safe_load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Expected list in YAML, got {type(raw_data)}")

        # Process each scenario
        all_examples = []
        for idx, item in enumerate(raw_data):
            # Validate required fields
            required_fields = ['scenario', 'question', 'answer_prefix',
                             'honest_completion', 'deceptive_completion']
            missing = [f for f in required_fields if f not in item]
            if missing:
                raise ValueError(f"Missing fields in item {idx}: {missing}")

            # Create neutral prompt (model generates without deception trigger)
            prompt = DECEPTION_ROLEPLAYING.format(
                scenario=item['scenario'],
                question=item['question'],
                answer_prefix=item['answer_prefix']
            )

            # Create example (no label yet - will be assigned after generation)
            example = {
                "prompt": prompt,
                "gold_label": None,  # Placeholder - assigned during caching
                "metadata": {
                    "dataset": "Deception-Roleplaying",
                    "id": f"roleplaying_{idx}",
                    "split": self.split,
                    "scenario": item['scenario'],
                    "question": item['question'],
                    "answer_prefix": item['answer_prefix'],
                    "honest_reference": item['honest_completion'],
                    "deceptive_reference": item['deceptive_completion']
                }
            }

            all_examples.append(example)

        # Create deterministic train/val/test splits
        random.seed(self.random_seed)
        random.shuffle(all_examples)

        n = len(all_examples)
        train_end = int(0.6 * n)
        val_end = int(0.8 * n)

        if self.split == "train":
            self.data = all_examples[:train_end]
        elif self.split == "validation":
            self.data = all_examples[train_end:val_end]
        elif self.split == "test":
            self.data = all_examples[val_end:]
        else:
            # If split is "all" or unknown, return everything
            self.data = all_examples

        # Apply limit if specified
        self.apply_limit()

        # Log summary
        print(f"Loaded {len(self.data)} examples for split '{self.split}'")
        print(f"  Total scenarios in YAML: {n}")
        print(f"  Train split: {train_end} ({100*train_end/n:.1f}%)")
        print(f"  Val split: {val_end - train_end} ({100*(val_end-train_end)/n:.1f}%)")
        print(f"  Test split: {n - val_end} ({100*(n-val_end)/n:.1f}%)")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single example"""
        return self.data[idx]


class DeceptionInsiderTradingDataset(BaseDataset):
    """
    Backward-compatible InsiderTrading loader that now delegates to SallyConcat.

    Legacy callers may still request the old InsiderTrading alias. This keeps
    those code paths working, but the loaded examples always come from the Sally
    concat dataset with ``gold_label`` / ``report_label`` semantics.
    """

    def __init__(self,
                 split: str = "train",
                 limit: Optional[int] = None,
                 data_file: str = "data/apollo_raw/insider_trading/sally_concat_examples.jsonl",
                 random_seed: int = 42,
                 include_ambiguous: bool = False):
        super().__init__(split=split, limit=limit)
        self.data_file = data_file
        self.random_seed = random_seed
        self.include_ambiguous = include_ambiguous

    def load_data(self):
        delegate = DeceptionInsiderTradingSallyConcatDataset(
            split=self.split,
            limit=self.limit,
            data_file=self.data_file,
        )
        delegate.load_data()
        self.data = delegate.data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class DeceptionInsiderTradingSallyConcatDataset(BaseDataset):
    """
    Insider Trading dataset loader for Sally-only concatenated prompts.

    Expects JSONL rows produced by:
        scripts/data/prepare_insider_trading_sally_dataset.py

    Row format:
        {
            "prompt": "...",
            "completion": "...",
            "gold_label": 0|1,
            "split": "train|validation|test",
            "metadata": {...}
        }
    """

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        data_file: str = "data/apollo_raw/insider_trading/sally_concat_examples.jsonl",
    ):
        super().__init__(split=split, limit=limit)
        self.data_file = data_file

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"Sally-concat dataset not found: {self.data_file}\n"
                "Create it first with:\n"
                "  python scripts/data/prepare_insider_trading_sally_dataset.py "
                "--input_jsonl <path/to/Insider_trading_final.jsonl>"
            )

    def load_data(self):
        rows: List[Dict[str, Any]] = []
        skipped = {"bad_json": 0, "missing_fields": 0, "bad_label": 0}

        with open(self.data_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped["bad_json"] += 1
                    continue

                if self.split in {"train", "validation", "test"}:
                    if row.get("split") != self.split:
                        continue

                prompt = row.get("prompt")
                completion = row.get("completion")
                if not isinstance(prompt, str) or not isinstance(completion, str):
                    skipped["missing_fields"] += 1
                    continue

                try:
                    gold_label = int(row.get("gold_label"))
                except (TypeError, ValueError):
                    skipped["bad_label"] += 1
                    continue

                if gold_label not in {0, 1}:
                    skipped["bad_label"] += 1
                    continue

                metadata = row.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata.setdefault("dataset", "Deception-InsiderTrading-SallyConcat")
                metadata.setdefault("split", row.get("split", self.split))
                metadata.setdefault("id", f"insider_sally_{len(rows)}")

                rows.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "gold_label": gold_label,
                        "metadata": metadata,
                    }
                )

        self.data = rows
        self.apply_limit()

        n_honest = sum(1 for r in self.data if r.get("gold_label") == 0)
        n_deceptive = sum(1 for r in self.data if r.get("gold_label") == 1)
        print(
            f"Loaded {len(self.data)} Sally-concat insider examples for split '{self.split}' "
            f"({n_honest} non-deceptive, {n_deceptive} deceptive)"
        )
        if any(v > 0 for v in skipped.values()):
            print(
                "  Skipped rows -> "
                f"bad_json: {skipped['bad_json']}, "
                f"missing_fields: {skipped['missing_fields']}, "
                f"bad_label: {skipped['bad_label']}"
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class DeceptionAILiarDataset(BaseDataset):
    """
    Apollo Research AILiar Dataset (pre-generated rollouts).

    This loader expects paired honest + deceptive outputs per rollout.
    If paired_examples.jsonl exists, it loads from that directly.
    Otherwise, it builds pairs on the fly from the raw rollouts JSON.

    Labels:
    - honest => 0
    - deceptive => 1
    """

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        data_file: str = "data/apollo_raw/ai_liar/ai_liar__original_without_answers__llama-70b.json",
        paired_examples_file: str = "data/apollo_raw/ai_liar/paired_examples.jsonl",
        random_seed: int = 42,
        max_pairs: int = 200,
    ):
        super().__init__(split=split, limit=limit)
        self.data_file = data_file
        self.paired_examples_file = paired_examples_file
        self.random_seed = random_seed
        self.max_pairs = max_pairs

        if not os.path.exists(self.data_file) and not os.path.exists(self.paired_examples_file):
            raise FileNotFoundError(
                f"AILiar data not found. Expected:\n"
                f"  - {self.data_file}\n"
                f"  - {self.paired_examples_file}\n"
            )

    def load_data(self):
        if os.path.exists(self.paired_examples_file):
            self.data = self._load_from_paired_examples()
        else:
            self.data = self._load_and_pair_raw()

        # Apply limit
        self.apply_limit()

    def _load_from_paired_examples(self) -> List[Dict[str, Any]]:
        rows = []
        with open(self.paired_examples_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != self.split:
                    continue
                meta = row.get("metadata")
                if isinstance(meta, dict):
                    meta.setdefault("has_pregenerated_response", True)
                    if "input_messages" not in meta and "input_messages" in row:
                        meta["input_messages"] = row.get("input_messages")
                rows.append(row)
        return rows

    def _load_and_pair_raw(self) -> List[Dict[str, Any]]:
        with open(self.data_file, "r") as f:
            raw = json.load(f)

        rollouts = raw.get("rollouts", [])
        paired = []
        for idx, rollout in enumerate(rollouts):
            deception_pct = rollout.get("deception_percentage")
            if deception_pct is None:
                continue
            if not (0 < float(deception_pct) <= 0.8):
                continue

            graded = rollout.get("graded_deception", [])
            if not graded:
                continue

            honest_idx = next((i for i, g in enumerate(graded) if str(g).lower() == "honest"), None)
            deceptive_idx = next((i for i, g in enumerate(graded) if str(g).lower() == "deceptive"), None)
            if honest_idx is None or deceptive_idx is None:
                continue

            outputs = rollout.get("output_strs")
            if not outputs:
                outputs = [m.get("content", "") for m in rollout.get("output_messages", [])]

            if honest_idx >= len(outputs) or deceptive_idx >= len(outputs):
                continue

            input_messages = rollout.get("input_messages", [])
            system_msg = next((m.get("content", "") for m in input_messages if m.get("role") == "system"), "")
            user_msgs = [m.get("content", "") for m in input_messages if m.get("role") == "user"]

            prompt_parts = []
            if system_msg:
                prompt_parts.append(system_msg.strip())
            if user_msgs:
                prompt_parts.append("\n\n".join(u.strip() for u in user_msgs if u.strip()))
            prompt = "\n\n".join(p for p in prompt_parts if p)

            paired.append(
                {
                    "rollout_id": f"ai_liar_{idx}",
                    "prompt": prompt,
                    "input_messages": input_messages,
                    "honest_output": outputs[honest_idx],
                    "deceptive_output": outputs[deceptive_idx],
                    "honest_idx": honest_idx,
                    "deceptive_idx": deceptive_idx,
                    "graded_deception": graded,
                    "deception_percentage": deception_pct,
                }
            )

        random.seed(self.random_seed)
        random.shuffle(paired)
        paired = paired[: self.max_pairs]

        # Rollout-level split
        n = len(paired)
        train_end = int(0.6 * n)
        val_end = int(0.8 * n)

        examples = []
        for i, row in enumerate(paired):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "validation"
            else:
                split = "test"
            if split != self.split:
                continue

            base_meta = {
                "dataset": "Deception-AILiar",
                "rollout_id": row["rollout_id"],
                "split": split,
                "input_messages": row["input_messages"],
                "graded_deception": row["graded_deception"],
                "deception_percentage": row["deception_percentage"],
                "has_pregenerated_response": True,
            }

            examples.append(
                {
                    "prompt": row["prompt"],
                    "completion": row["honest_output"],
                    "gold_label": 0,
                    "metadata": {**base_meta, "id": f"{row['rollout_id']}_honest", "output_index": row["honest_idx"]},
                }
            )
            examples.append(
                {
                    "prompt": row["prompt"],
                    "completion": row["deceptive_output"],
                    "gold_label": 1,
                    "metadata": {**base_meta, "id": f"{row['rollout_id']}_deceptive", "output_index": row["deceptive_idx"]},
                }
            )

        return examples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single example"""
        return self.data[idx]


class DeceptionTypedMessagesDataset(BaseDataset):
    """
    Loader for instructed-deception data with pre-generated assistant completions.

    Expected row fields (parquet/jsonl):
    - messages or messages_clean: list[{role, content}] with system/user/assistant
    - label and/or deceptive: binary label (1=deceptive, 0=non-deceptive)
    - optional metadata: id/index/dataset/subdataset/model/temperature/...
    """

    TAXONOMY_KEYS = (
        "deception_type_primary",
        "deception_types_mentioned",
        "classification_reasoning",
        "classification_confidence",
        "classification_raw_response",
    )

    DATASET_MARKERS = {
        "Deception-InstructedDeception": ("instructed-deception", "instructed_deception"),
        "Deception-Mask": ("mask__", "mask"),
        "Deception-HarmPressureChoice": ("harm-pressure-choice", "harm_pressure_choice"),
        "Deception-ConvincingGame": ("convincing-game", "convincing_game"),
        "Deception-ClaimsDefinitional": ("claims_definitional", "claims-definitional", "claimsdefinitional"),
        "Deception-ClaimsEvidential": ("claims_evidential", "claims-evidential", "claimsevidential"),
        "Deception-ClaimsFictional": ("claims_fictional", "claims-fictional", "claimsfictional"),
        "Deception-ClaimsLogical": ("claims_logical", "claims-logical", "claimslogical"),
    }

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        data_file: str = "data/apollo_raw/instructed_deception/instructed-deception_gemma_2_9b_it_completions_deception_typed.parquet",
        random_seed: int = 42,
        dataset_name: Optional[str] = None,
        id_prefix: Optional[str] = None,
    ):
        super().__init__(split=split, limit=limit)
        self.data_file = data_file
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        self.id_prefix = id_prefix

        possible_paths = self._candidate_paths(data_file, dataset_name)

        self.actual_data_file = None
        for path in possible_paths:
            if os.path.exists(path):
                self.actual_data_file = path
                break

        if self.actual_data_file is None:
            raise FileNotFoundError(
                "Typed deception dataset not found. Tried:\n"
                + "\n".join(f"  - {p}" for p in possible_paths)
            )

        self._validate_resolved_data_file()

        # If not explicitly provided, infer metadata dataset/id prefix from source file.
        inferred_source = os.path.basename(self.actual_data_file).lower()
        if self.dataset_name is None:
            if "mask__" in inferred_source:
                self.dataset_name = "Deception-Mask"
            elif "claims_definitional" in inferred_source:
                self.dataset_name = "Deception-ClaimsDefinitional"
            elif "claims_evidential" in inferred_source:
                self.dataset_name = "Deception-ClaimsEvidential"
            elif "claims_fictional" in inferred_source:
                self.dataset_name = "Deception-ClaimsFictional"
            elif "claims_logical" in inferred_source:
                self.dataset_name = "Deception-ClaimsLogical"
            elif "harm-pressure-choice__" in inferred_source:
                self.dataset_name = "Deception-HarmPressureChoice"
            elif "convincing-game__" in inferred_source:
                self.dataset_name = "Deception-ConvincingGame"
            else:
                self.dataset_name = "Deception-InstructedDeception"
        if self.id_prefix is None:
            if "mask__" in inferred_source:
                self.id_prefix = "mask"
            elif "claims_definitional" in inferred_source:
                self.id_prefix = "claims_definitional"
            elif "claims_evidential" in inferred_source:
                self.id_prefix = "claims_evidential"
            elif "claims_fictional" in inferred_source:
                self.id_prefix = "claims_fictional"
            elif "claims_logical" in inferred_source:
                self.id_prefix = "claims_logical"
            elif "harm-pressure-choice__" in inferred_source:
                self.id_prefix = "harm_pressure_choice"
            elif "convincing-game__" in inferred_source:
                self.id_prefix = "convincing_game"
            else:
                self.id_prefix = "instructed_deception"

    @classmethod
    def _candidate_paths(cls, data_file: str, dataset_name: Optional[str]) -> List[str]:
        def _dedupe(paths: List[str]) -> List[str]:
            seen = set()
            ordered = []
            for path in paths:
                if path in seen:
                    continue
                seen.add(path)
                ordered.append(path)
            return ordered

        downloads = os.path.join(os.path.expanduser("~"), "Downloads")

        instructed_paths = [
            data_file,
            "data/apollo_raw/instructed-deception/instructed-deception__gemma_2_9b_it__deception_typed__messages_clean.jsonl",
            "data/apollo_raw/instructed-deception/instructed-deception_clean.jsonl",
            "data/apollo_raw/instructed_deception/instructed-deception_gemma_2_9b_it_completions_deception_typed.parquet",
            "data/apollo_raw/instructed_deception/instructed-deception_gemma_2_9b_it_completions_deception_typed.jsonl",
            "data/apollo_raw/instructed_deception/instructed-deception_clean.jsonl",
            "data/apollo_raw/instructed-deception_gemma_2_9b_it_completions_deception_typed.parquet",
            "data/apollo_raw/instructed-deception_gemma_2_9b_it_completions_deception_typed.jsonl",
            os.path.join(
                downloads,
                "instructed-deception__gemma_2_9b_it__deception_typed__messages_clean.jsonl",
            ),
        ]
        mask_paths = [
            data_file,
            "data/apollo_raw/mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl",
            "data/apollo_raw/mask/mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl",
            "data/apolloraw/mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl",
            os.path.join(
                downloads,
                "mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl",
            ),
        ]
        claims_paths = {
            "Deception-ClaimsDefinitional": [
                data_file,
                "data/apollo_raw/claims_definitional/claims_definitional__truth_spec__messages_clean.jsonl",
            ],
            "Deception-ClaimsEvidential": [
                data_file,
                "data/apollo_raw/claims_evidential/claims_evidential__truth_spec__messages_clean.jsonl",
            ],
            "Deception-ClaimsFictional": [
                data_file,
                "data/apollo_raw/claims_fictional/claims_fictional__truth_spec__messages_clean.jsonl",
            ],
            "Deception-ClaimsLogical": [
                data_file,
                "data/apollo_raw/claims_logical/claims_logical__truth_spec__messages_clean.jsonl",
            ],
        }
        generic_paths = instructed_paths + mask_paths

        if dataset_name == "Deception-InstructedDeception":
            return _dedupe(instructed_paths)
        if dataset_name == "Deception-Mask":
            return _dedupe(mask_paths)
        if dataset_name in claims_paths:
            return _dedupe(claims_paths[dataset_name])
        return _dedupe(generic_paths)

    def _validate_resolved_data_file(self) -> None:
        source = os.path.basename(self.actual_data_file).lower()
        if self.dataset_name == "Deception-InstructedDeception":
            if any(marker in source for marker in self.DATASET_MARKERS["Deception-Mask"]):
                raise ValueError(
                    "Resolved Deception-InstructedDeception to a Mask file. "
                    f"Refusing to load cross-dataset fallback: {self.actual_data_file}"
                )
        if self.dataset_name == "Deception-Mask":
            instructed_markers = self.DATASET_MARKERS["Deception-InstructedDeception"]
            if any(marker in source for marker in instructed_markers) and "mask__" not in source:
                raise ValueError(
                    "Resolved Deception-Mask to an instructed-deception file. "
                    f"Refusing to load cross-dataset fallback: {self.actual_data_file}"
                )

    @classmethod
    def _classify_dataset_hint(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        for dataset_name, markers in cls.DATASET_MARKERS.items():
            if any(marker in text for marker in markers):
                return dataset_name
        return None

    def _validate_row_dataset(self, row: Dict[str, Any], base_id: Any) -> Optional[str]:
        if self.dataset_name not in self.DATASET_MARKERS:
            return None

        hints = [
            ("dataset", row.get("dataset")),
            ("subdataset", row.get("subdataset")),
            ("id", base_id),
        ]
        for key, value in hints:
            hinted_dataset = self._classify_dataset_hint(value)
            if hinted_dataset is not None and hinted_dataset != self.dataset_name:
                return f"{key}={value!r} suggests {hinted_dataset}"
        return None

    @staticmethod
    def _normalize_messages(raw_messages: Any) -> List[Dict[str, str]]:
        if raw_messages is None:
            return []

        # Handle numpy arrays / pyarrow lists
        if hasattr(raw_messages, "tolist") and not isinstance(raw_messages, list):
            raw_messages = raw_messages.tolist()

        # Handle stringified lists
        if isinstance(raw_messages, str):
            parsed = None
            txt = raw_messages.strip()
            if not txt:
                return []
            try:
                parsed = json.loads(txt)
            except Exception:
                try:
                    parsed = ast.literal_eval(txt)
                except Exception:
                    return []
            raw_messages = parsed

        if isinstance(raw_messages, dict):
            raw_messages = [raw_messages]

        if not isinstance(raw_messages, list):
            return []

        normalized = []
        for msg in raw_messages:
            if hasattr(msg, "tolist") and not isinstance(msg, dict):
                msg = msg.tolist()
            if not isinstance(msg, dict):
                continue

            role = str(msg.get("role", "")).strip().lower()
            content = msg.get("content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = str(content)

            if role in {"system", "user", "assistant"}:
                normalized.append({"role": role, "content": content})

        return normalized

    @staticmethod
    def _strip_system_prefix_from_first_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Conservative cleaning: if first user content starts exactly with the
        system content, remove that prefix once.
        """
        if not messages:
            return messages

        system_text = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_text = (msg.get("content") or "").strip()
                break

        if not system_text:
            return messages

        cleaned = []
        stripped_once = False
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user" and not stripped_once:
                candidate = content.lstrip()
                if candidate.startswith(system_text):
                    new_content = candidate[len(system_text):].lstrip("\n\r\t ")
                    content = new_content
                    stripped_once = True
            cleaned.append({"role": role, "content": content})
        return cleaned

    @staticmethod
    def _coerce_binary_label(raw: Any) -> Optional[int]:
        if isinstance(raw, bool):
            return 1 if raw else 0
        if isinstance(raw, (int, float)):
            val = int(raw)
            return val if val in {0, 1} else None
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in {"1", "true", "yes", "deceptive", "negative"}:
                return 1
            if s in {"0", "false", "no", "honest", "positive", "non_deceptive"}:
                return 0
        return None

    @classmethod
    def _parse_label(cls, row: Dict[str, Any]) -> Optional[int]:
        # Canonical label for newer deception-typed datasets is `label` (0/1).
        for key in ("label", "deceptive", "gold_label"):
            if key in row and row.get(key) is not None:
                coerced = cls._coerce_binary_label(row.get(key))
                if coerced is not None:
                    return coerced
        return None

    @staticmethod
    def _normalize_split_name(value: Any) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip().lower()
        if s in {"train", "training"}:
            return "train"
        if s in {"validation", "val", "valid", "dev"}:
            return "validation"
        if s in {"test", "eval", "evaluation"}:
            return "test"
        return None

    def _load_rows(self) -> List[Dict[str, Any]]:
        path = self.actual_data_file
        lower = path.lower()

        if lower.endswith(".jsonl"):
            rows = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows

        if lower.endswith(".json"):
            with open(path, "r") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                return raw
            if isinstance(raw, dict):
                if isinstance(raw.get("data"), list):
                    return raw["data"]
                if isinstance(raw.get("rows"), list):
                    return raw["rows"]
            raise ValueError(f"Unsupported JSON structure in {path}")

        if lower.endswith(".parquet"):
            try:
                from datasets import load_dataset
                ds = load_dataset("parquet", data_files=path, split="train")
                return [dict(ds[i]) for i in range(len(ds))]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read parquet file {path}. "
                    "Install parquet dependencies (pyarrow) or convert to jsonl."
                ) from e

        raise ValueError(f"Unsupported file type for instructed-deception dataset: {path}")

    def _assign_splits(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stratified 60/20/20 split by label for deterministic train/val/test.
        """
        groups = {0: [], 1: []}
        for ex in examples:
            lbl = int(ex.get("gold_label", 0))
            groups.setdefault(lbl, []).append(ex)

        rng = random.Random(self.random_seed)
        split_rows = []
        for lbl, rows in groups.items():
            rng.shuffle(rows)
            n = len(rows)
            if n == 1:
                train_end = 1
                val_end = 1
            elif n == 2:
                train_end = 1
                val_end = 2
            else:
                train_end = max(1, int(0.6 * n))
                val_end = min(n, max(train_end + 1, int(0.8 * n)))
            for i, row in enumerate(rows):
                if i < train_end:
                    row_split = "train"
                elif i < val_end:
                    row_split = "validation"
                else:
                    row_split = "test"
                row["split"] = row_split
                split_rows.append(row)
        rng.shuffle(split_rows)
        return split_rows

    def load_data(self):
        raw_rows = self._load_rows()
        if not raw_rows:
            raise ValueError(f"No rows found in instructed-deception data: {self.actual_data_file}")

        processed = []
        skipped = {"bad_messages": 0, "missing_assistant": 0, "missing_label": 0}
        conflicting_rows = []

        for idx, row in enumerate(raw_rows):
            has_messages_clean = row.get("messages_clean") is not None
            messages_raw = row.get("messages_clean") if has_messages_clean else row.get("messages")
            messages = self._normalize_messages(messages_raw)
            if not messages:
                skipped["bad_messages"] += 1
                continue

            # Keep fallback cleanup only when explicit cleaned messages are unavailable.
            if not has_messages_clean:
                messages = self._strip_system_prefix_from_first_user(messages)

            assistant_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "assistant":
                    assistant_idx = i
                    break
            if assistant_idx is None:
                skipped["missing_assistant"] += 1
                continue

            completion = messages[assistant_idx].get("content", "")
            label = self._parse_label(row)
            if label is None:
                skipped["missing_label"] += 1
                continue

            prompt_messages = [m for m in messages[:assistant_idx] if m.get("role") in {"system", "user"}]
            if not prompt_messages:
                prompt_messages = [m for m in messages if m.get("role") in {"system", "user"}]
            prompt = "\n\n".join((m.get("content", "") or "").strip() for m in prompt_messages if (m.get("content", "") or "").strip())

            base_id = row.get("id", row.get("index", row.get("original_index", idx)))
            dataset_conflict = self._validate_row_dataset(row, base_id)
            if dataset_conflict is not None:
                conflicting_rows.append({"index": idx, "reason": dataset_conflict})
                continue
            example_id = f"{self.id_prefix}_{base_id}"

            split_name = self._normalize_split_name(row.get("split"))

            metadata = {
                "dataset": self.dataset_name,
                "id": example_id,
                "split": split_name or self.split,
                "source_id": base_id,
                "input_messages": prompt_messages,
                "has_pregenerated_response": True,
            }

            for key in (
                "subdataset",
                "dataset",
                "dataset_index",
                "model",
                "temperature",
                "original_index",
            ):
                if key in row:
                    if key == "dataset":
                        metadata["source_dataset"] = row.get(key)
                    else:
                        metadata[key] = row.get(key)

            for key in self.TAXONOMY_KEYS:
                if key in row:
                    metadata[key] = row.get(key)

            processed.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "gold_label": label,
                    "metadata": metadata,
                    "_explicit_split": split_name,
                }
            )

        if conflicting_rows:
            preview = "; ".join(
                f"row {item['index']}: {item['reason']}"
                for item in conflicting_rows[:5]
            )
            raise ValueError(
                f"Found {len(conflicting_rows)} rows inconsistent with {self.dataset_name}. "
                f"Examples: {preview}"
            )

        if not processed:
            raise ValueError(
                "No valid instructed-deception examples after preprocessing. "
                f"Skipped: {skipped}"
            )

        has_explicit_splits = any(ex.get("_explicit_split") for ex in processed)
        if has_explicit_splits:
            explicit_examples = []
            missing_split_examples = []
            for ex in processed:
                ex_split = ex.get("_explicit_split")
                if ex_split is None:
                    missing_split_examples.append(ex)
                    continue
                ex["split"] = ex_split
                ex["metadata"]["split"] = ex_split
                explicit_examples.append(ex)

            if missing_split_examples:
                inferred = self._assign_splits(missing_split_examples)
                for ex in inferred:
                    ex["metadata"]["split"] = ex.get("split", self.split)
                split_examples = explicit_examples + inferred
            else:
                split_examples = explicit_examples
        else:
            split_examples = self._assign_splits(processed)
            for ex in split_examples:
                ex["metadata"]["split"] = ex.get("split", self.split)

        if self.split in {"train", "validation", "test"}:
            self.data = [ex for ex in split_examples if ex.get("split") == self.split]
        else:
            self.data = split_examples

        for ex in self.data:
            ex.pop("_explicit_split", None)

        self.apply_limit()

        n_honest = sum(1 for ex in self.data if int(ex.get("gold_label", -1)) == 0)
        n_deceptive = sum(1 for ex in self.data if int(ex.get("gold_label", -1)) == 1)
        print(
            f"Loaded {len(self.data)} {self.dataset_name} examples for split '{self.split}' "
            f"({n_honest} honest, {n_deceptive} deceptive)"
        )
        if any(v > 0 for v in skipped.values()):
            print(
                "  Skipped rows -> "
                f"bad_messages: {skipped['bad_messages']}, "
                f"missing_assistant: {skipped['missing_assistant']}, "
                f"missing_label: {skipped['missing_label']}"
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class DeceptionMaskDataset(DeceptionTypedMessagesDataset):
    """
    Mask dataset wrapper that reuses instructed-deception parsing logic.
    """

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        data_file: str = "data/apollo_raw/mask__gemma_2_9b_it__deception_typed__messages_clean.jsonl",
        random_seed: int = 42,
    ):
        super().__init__(
            split=split,
            limit=limit,
            data_file=data_file,
            random_seed=random_seed,
            dataset_name="Deception-Mask",
            id_prefix="mask",
        )


class DeceptionInstructedDeceptionDataset(DeceptionTypedMessagesDataset):
    """
    Explicit instructed-deception dataset wrapper with fail-closed path resolution.
    """

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        data_file: str = "data/apollo_raw/instructed-deception/instructed-deception__gemma_2_9b_it__deception_typed__messages_clean.jsonl",
        random_seed: int = 42,
    ):
        super().__init__(
            split=split,
            limit=limit,
            data_file=data_file,
            random_seed=random_seed,
            dataset_name="Deception-InstructedDeception",
            id_prefix="instructed_deception",
        )


class DeceptionClaimsDefinitionalDataset(DeceptionTypedMessagesDataset):
    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        data_file: str = "data/apollo_raw/claims_definitional/claims_definitional__truth_spec__messages_clean.jsonl",
        random_seed: int = 42,
    ):
        super().__init__(
            split=split,
            limit=limit,
            data_file=data_file,
            random_seed=random_seed,
            dataset_name="Deception-ClaimsDefinitional",
            id_prefix="claims_definitional",
        )


class DeceptionClaimsEvidentialDataset(DeceptionTypedMessagesDataset):
    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        data_file: str = "data/apollo_raw/claims_evidential/claims_evidential__truth_spec__messages_clean.jsonl",
        random_seed: int = 42,
    ):
        super().__init__(
            split=split,
            limit=limit,
            data_file=data_file,
            random_seed=random_seed,
            dataset_name="Deception-ClaimsEvidential",
            id_prefix="claims_evidential",
        )


class DeceptionClaimsFictionalDataset(DeceptionTypedMessagesDataset):
    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        data_file: str = "data/apollo_raw/claims_fictional/claims_fictional__truth_spec__messages_clean.jsonl",
        random_seed: int = 42,
    ):
        super().__init__(
            split=split,
            limit=limit,
            data_file=data_file,
            random_seed=random_seed,
            dataset_name="Deception-ClaimsFictional",
            id_prefix="claims_fictional",
        )


class DeceptionClaimsLogicalDataset(DeceptionTypedMessagesDataset):
    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        data_file: str = "data/apollo_raw/claims_logical/claims_logical__truth_spec__messages_clean.jsonl",
        random_seed: int = 42,
    ):
        super().__init__(
            split=split,
            limit=limit,
            data_file=data_file,
            random_seed=random_seed,
            dataset_name="Deception-ClaimsLogical",
            id_prefix="claims_logical",
        )
