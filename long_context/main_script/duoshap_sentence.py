#!/usr/bin/env python3
"""
Global sentence-level DuoShap for selected LongBench examples.

Supported tasks:
  - passage_retrieval_en
  - hotpotqa
  - 2wikimqa

Core logic:
  - players = all individual sentences across the full context
  - one sentence = one DuoShap player
  - preserve paragraph/passage headers
  - omitted sentences are replaced by OMIT_SENTENCE_TOKEN, default ""
  - value function = avg log p(gold answer | masked context prompt)
  - per-player conditioned DuoShap
  - antithetic subset that still contains player i
  - deterministic per-player RNG
  - save raw phi per sentence shard

New feature:
  - optional --selected-indices-json
  - JSON format:
      {
        "passage_retrieval_en": [127, 155, ...],
        "hotpotqa": [28, 7, ...],
        "2wikimqa": [20, 58, ...]
      }

Slurm array mapping:
  selected_indices = indices for current task
  blocks_per_example = ceil(MAX_SENTENCE_SHARDS / SENTENCE_SHARDS_PER_JOB)

  array_id 0 -> selected_indices[0], shard block 0
  array_id 1 -> selected_indices[0], shard block 1
  ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG
# Most values can be overridden from Slurm command line.
# ============================================================

@dataclass
class Config:
    HF_HOME: Path = Path("xxxxxxxxxxxxx")

    LONGBENCH_DATA_DIR: Path = Path(
        "xxxx"
    )

    OUT_DIR: Path = Path(
        "xxxxxxxx"
    )

    RUN_NAME: str = "global_sentence_true_selected_qwen25_7b"

    # Previous paragraph-level DuoShap output.
    PARAGRAPH_OUT_ROOT: Path = Path(
        "xxxxxxxx"
    )
    PARAGRAPH_RUN_NAME: str = "duoshap_qwen25_7b_paragraph_level"

    TASK_NAME: str = "passage_retrieval_en"

    # Optional selected-index file. If provided, this overrides normal contiguous selection.
    SELECTED_INDICES_JSON: Optional[Path] = None

    EXAMPLE_START: int = 0
    EXAMPLE_END: Optional[int] = None
    MAX_EXAMPLES: Optional[int] = 1

    # Slurm array mapping:
    # array_id -> selected example offset + sentence shard block.
    ARRAY_ID: int = 0
    MAX_SENTENCE_SHARDS: int = 60
    SENTENCE_SHARDS_PER_JOB: int = 20
    SENTENCE_SHARD_SIZE: int = 10

    MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"
    DTYPE: str = "fp16"
    USE_CHAT_TEMPLATE: bool = True
    USE_MODEL_CACHE: bool = False

    # Attention backend:
    #   flash_attention_2 = fastest on supported GPUs (Ampere/Ada/Hopper; not V100)
    #   sdpa              = PyTorch scaled-dot-product attention fallback
    #   eager             = vanilla attention fallback
    ATTENTION_IMPL: str = "flash_attention_2"

    # Number of coalition prompts evaluated in one forward pass.
    # For 7B + long context on 32GB V100, start with 1 or 2.
    # On A100/H100, try 2, 4, or 8 after a debug run.
    VALUE_BATCH_SIZE: int = 2

    SEED: int = 42

    OMIT_UNIT_TOKEN: str = "[OMITTED]"
    OMIT_SENTENCE_TOKEN: str = ""

    ENABLE_VALUE_CACHE: bool = True
    SAVE_VALUE_CACHE: bool = False
    MAX_PROMPT_TOKENS_WARN: int = 40000

    NUM_GROUPS_PER_FEATURE: int = 5
    USE_ANTITHETIC: bool = True
    EXCLUDE_FULL_SET_IN_K: bool = True

    SKIP_IF_EXISTS: bool = True

    # QA support-label settings for PR/ROC later.
    QA_MIN_ANSWER_CHARS_FOR_SUPPORT: int = 3
    QA_EXCLUDE_YES_NO: bool = True


# ============================================================
# ARGUMENTS
# ============================================================

def parse_optional_int(x: str) -> Optional[int]:
    if x is None:
        return None
    if str(x).lower() in {"none", "null", "-1"}:
        return None
    return int(x)


def parse_optional_path(x: str) -> Optional[Path]:
    if x is None:
        return None
    if str(x).lower() in {"none", "null", ""}:
        return None
    return Path(x)


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser()

    p.add_argument(
        "--task",
        type=str,
        default=cfg.TASK_NAME,
        choices=["passage_retrieval_en", "hotpotqa", "2wikimqa"],
    )
    p.add_argument("--run-name", type=str, default=cfg.RUN_NAME)

    p.add_argument(
        "--selected-indices-json",
        type=parse_optional_path,
        default=cfg.SELECTED_INDICES_JSON,
        help="Optional JSON file mapping task name to selected example indices.",
    )

    p.add_argument("--hf-home", type=Path, default=cfg.HF_HOME)
    p.add_argument("--data-dir", type=Path, default=cfg.LONGBENCH_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=cfg.OUT_DIR)

    p.add_argument("--paragraph-out-root", type=Path, default=cfg.PARAGRAPH_OUT_ROOT)
    p.add_argument("--paragraph-run-name", type=str, default=cfg.PARAGRAPH_RUN_NAME)

    p.add_argument("--model-id", type=str, default=cfg.MODEL_ID)
    p.add_argument("--dtype", type=str, default=cfg.DTYPE, choices=["fp16", "fp32"])
    p.add_argument(
        "--attention-impl",
        type=str,
        default=cfg.ATTENTION_IMPL,
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Use flash_attention_2 on supported GPUs; use sdpa/eager as fallback.",
    )
    p.add_argument(
        "--value-batch-size",
        type=int,
        default=cfg.VALUE_BATCH_SIZE,
        help="Number of coalition prompts evaluated per model forward.",
    )

    p.add_argument("--example-start", type=int, default=cfg.EXAMPLE_START)
    p.add_argument("--example-end", type=parse_optional_int, default=cfg.EXAMPLE_END)
    p.add_argument("--max-examples", type=parse_optional_int, default=cfg.MAX_EXAMPLES)

    p.add_argument(
        "--array-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", cfg.ARRAY_ID)),
    )
    p.add_argument("--max-sentence-shards", type=int, default=cfg.MAX_SENTENCE_SHARDS)
    p.add_argument("--sentence-shards-per-job", type=int, default=cfg.SENTENCE_SHARDS_PER_JOB)
    p.add_argument("--sentence-shard-size", type=int, default=cfg.SENTENCE_SHARD_SIZE)

    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--num-groups-per-feature", type=int, default=cfg.NUM_GROUPS_PER_FEATURE)

    p.add_argument("--no-antithetic", action="store_true")
    p.add_argument("--include-full-set-in-k", action="store_true")
    p.add_argument("--no-value-cache", action="store_true")
    p.add_argument("--save-value-cache", action="store_true")
    p.add_argument("--no-skip-if-exists", action="store_true")

    args = p.parse_args()

    cfg.TASK_NAME = args.task
    cfg.RUN_NAME = args.run_name
    cfg.SELECTED_INDICES_JSON = args.selected_indices_json

    cfg.HF_HOME = args.hf_home
    cfg.LONGBENCH_DATA_DIR = args.data_dir
    cfg.OUT_DIR = args.out_dir

    cfg.PARAGRAPH_OUT_ROOT = args.paragraph_out_root
    cfg.PARAGRAPH_RUN_NAME = args.paragraph_run_name

    cfg.MODEL_ID = args.model_id
    cfg.DTYPE = args.dtype
    cfg.ATTENTION_IMPL = args.attention_impl
    cfg.VALUE_BATCH_SIZE = args.value_batch_size

    cfg.EXAMPLE_START = args.example_start
    cfg.EXAMPLE_END = args.example_end
    cfg.MAX_EXAMPLES = args.max_examples

    cfg.ARRAY_ID = args.array_id
    cfg.MAX_SENTENCE_SHARDS = args.max_sentence_shards
    cfg.SENTENCE_SHARDS_PER_JOB = args.sentence_shards_per_job
    cfg.SENTENCE_SHARD_SIZE = args.sentence_shard_size

    cfg.SEED = args.seed
    cfg.NUM_GROUPS_PER_FEATURE = args.num_groups_per_feature

    cfg.USE_ANTITHETIC = not args.no_antithetic
    cfg.EXCLUDE_FULL_SET_IN_K = not args.include_full_set_in_k
    cfg.ENABLE_VALUE_CACHE = not args.no_value_cache
    cfg.SAVE_VALUE_CACHE = args.save_value_cache
    cfg.SKIP_IF_EXISTS = not args.no_skip_if_exists

    return cfg


# ============================================================
# GENERAL HELPERS
# ============================================================

def configure_environment(cfg: Config) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["HF_HOME"] = str(cfg.HF_HOME)
    os.environ["HF_DATASETS_CACHE"] = str(cfg.HF_HOME / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cfg.HF_HOME / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cfg.HF_HOME)
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    cfg.HF_HOME.mkdir(parents=True, exist_ok=True)
    (cfg.HF_HOME / "datasets").mkdir(parents=True, exist_ok=True)
    (cfg.HF_HOME / "hub").mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def to_jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def write_json_once(path: Path, obj) -> None:
    if path.exists():
        return
    write_json(path, obj)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def select_indices(n: int, cfg: Config) -> List[int]:
    """
    Select example indices.

    If cfg.SELECTED_INDICES_JSON is provided:
      load selected indices for cfg.TASK_NAME from JSON,
      preserve the order in the JSON,
      optionally filter by EXAMPLE_START/EXAMPLE_END,
      optionally truncate by MAX_EXAMPLES.

    Otherwise:
      use contiguous range [EXAMPLE_START, EXAMPLE_END), then MAX_EXAMPLES.
    """
    if cfg.SELECTED_INDICES_JSON is not None:
        path = cfg.SELECTED_INDICES_JSON
        if not path.exists():
            raise FileNotFoundError(f"Missing selected indices JSON: {path}")

        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            if cfg.TASK_NAME not in obj:
                raise KeyError(
                    f"Task {cfg.TASK_NAME} not found in selected indices JSON. "
                    f"Available keys: {list(obj.keys())}"
                )
            raw_indices = obj[cfg.TASK_NAME]
        elif isinstance(obj, list):
            raw_indices = obj
        else:
            raise ValueError(
                "Selected indices JSON must be either a dict task->list[int] or a list[int]."
            )

        indices = [int(x) for x in raw_indices]

        bad = [i for i in indices if i < 0 or i >= n]
        if bad:
            raise ValueError(
                f"Selected indices out of range for task={cfg.TASK_NAME}, n={n}: {bad}"
            )

        start = max(0, cfg.EXAMPLE_START)
        end = cfg.EXAMPLE_END if cfg.EXAMPLE_END is not None else n
        end = min(end, n)
        indices = [i for i in indices if start <= i < end]

        if cfg.MAX_EXAMPLES is not None:
            indices = indices[: cfg.MAX_EXAMPLES]

        print(
            f"[INFO] Loaded {len(indices)} selected examples for task={cfg.TASK_NAME} "
            f"from {path}: {indices}",
            flush=True,
        )
        return indices

    start = max(0, cfg.EXAMPLE_START)
    end = cfg.EXAMPLE_END if cfg.EXAMPLE_END is not None else n
    end = min(end, n)

    indices = list(range(start, end))
    if cfg.MAX_EXAMPLES is not None:
        indices = indices[: cfg.MAX_EXAMPLES]

    return indices


def map_array_id_to_work(
    cfg: Config,
    selected_indices: Sequence[int],
) -> Tuple[Optional[int], Optional[List[int]], dict]:
    """
    Maps Slurm ARRAY_ID to:
      - one selected example_index
      - one or more sentence_shard_ids for that example

    blocks_per_example = ceil(MAX_SENTENCE_SHARDS / SENTENCE_SHARDS_PER_JOB)

    array_id:
      example_offset = array_id // blocks_per_example
      block_id       = array_id % blocks_per_example
      shard ids      = block_id * SENTENCE_SHARDS_PER_JOB ...
    """
    if cfg.SENTENCE_SHARDS_PER_JOB < 1:
        raise ValueError("SENTENCE_SHARDS_PER_JOB must be >= 1.")

    blocks_per_example = math.ceil(cfg.MAX_SENTENCE_SHARDS / cfg.SENTENCE_SHARDS_PER_JOB)

    example_offset = cfg.ARRAY_ID // blocks_per_example
    block_id = cfg.ARRAY_ID % blocks_per_example

    meta = {
        "array_id": cfg.ARRAY_ID,
        "blocks_per_example": blocks_per_example,
        "example_offset": example_offset,
        "block_id": block_id,
        "max_sentence_shards": cfg.MAX_SENTENCE_SHARDS,
        "sentence_shards_per_job": cfg.SENTENCE_SHARDS_PER_JOB,
    }

    if example_offset >= len(selected_indices):
        return None, None, meta

    example_index = int(selected_indices[example_offset])

    start_shard = block_id * cfg.SENTENCE_SHARDS_PER_JOB
    end_shard = min(start_shard + cfg.SENTENCE_SHARDS_PER_JOB, cfg.MAX_SENTENCE_SHARDS)
    shard_ids = list(range(start_shard, end_shard))

    return example_index, shard_ids, meta


def stable_player_seed(base_seed: int, task_name: str, example_index: int, player_i: int) -> int:
    task_hash = sum((i + 1) * ord(c) for i, c in enumerate(task_name))
    return int(
        (base_seed + 1009 * example_index + 9173 * task_hash + 10007 * player_i)
        % (2**32 - 1)
    )


# ============================================================
# TASK SPECS
# ============================================================

@dataclass(frozen=True)
class TaskSpec:
    name: str
    unit_name: str
    header_regex: re.Pattern
    expected_units: Optional[int]
    metric_type: str
    build_prompt: Callable[[str, str], str]


_PAR_HDR = re.compile(r"(Paragraph\s+(\d+)\s*:)", flags=re.IGNORECASE)
_PASSAGE_HDR = re.compile(r"(Passage\s+(\d+)\s*:)", flags=re.IGNORECASE)


def build_prompt_passage_retrieval(context: str, abstract: str) -> str:
    return (
        "Here are 30 paragraphs and an abstract. Determine which paragraph the abstract is from.\n"
        "Output the answer in the format \"Paragraph k\" where k is from 1 to 30.\n\n"
        f"{context}\n\n"
        f"Abstract:\n{abstract}\n\n"
        "Answer:"
    )


def build_prompt_qa(context: str, question: str) -> str:
    return (
        "Answer the question based ONLY on the given passages. Do not use outside knowledge.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


TASKS: Dict[str, TaskSpec] = {
    "passage_retrieval_en": TaskSpec(
        name="passage_retrieval_en",
        unit_name="Paragraph",
        header_regex=_PAR_HDR,
        expected_units=30,
        metric_type="gold_paragraph",
        build_prompt=build_prompt_passage_retrieval,
    ),
    "hotpotqa": TaskSpec(
        name="hotpotqa",
        unit_name="Passage",
        header_regex=_PASSAGE_HDR,
        expected_units=None,
        metric_type="answer_containment",
        build_prompt=build_prompt_qa,
    ),
    "2wikimqa": TaskSpec(
        name="2wikimqa",
        unit_name="Passage",
        header_regex=_PASSAGE_HDR,
        expected_units=None,
        metric_type="answer_containment",
        build_prompt=build_prompt_qa,
    ),
}


# ============================================================
# UNIT / SENTENCE PARSING
# ============================================================

def split_numbered_units(context: str, task: TaskSpec, omit_token: str) -> List[str]:
    matches = list(task.header_regex.finditer(context))

    if matches:
        chunks: Dict[int, str] = {}

        for idx, m in enumerate(matches):
            k = int(m.group(2))
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(context)
            chunks[k] = context[start:end].strip()

        max_k = task.expected_units if task.expected_units is not None else max(chunks.keys())

        units = []
        for k in range(1, max_k + 1):
            units.append(chunks.get(k, f"{task.unit_name} {k}: {omit_token}"))
        return units

    parts = [p.strip() for p in re.split(r"\n\s*\n", context) if p.strip()]
    if task.expected_units is not None:
        parts = parts[: task.expected_units]

    units = [f"{task.unit_name} {i + 1}: {p}" for i, p in enumerate(parts)]

    if task.expected_units is not None:
        while len(units) < task.expected_units:
            k = len(units) + 1
            units.append(f"{task.unit_name} {k}: {omit_token}")

    return units


def parse_unit_header(unit_text: str, task: TaskSpec, unit_k: int) -> Tuple[str, str]:
    pat = rf"^({task.unit_name}\s+\d+\s*:)\s*(.*)$"
    m = re.match(pat, unit_text.strip(), flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return f"{task.unit_name} {unit_k}:", unit_text.strip()
    return m.group(1), m.group(2).strip()


_ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "u.s.", "u.k.", "no.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.",
    "oct.", "nov.", "dec.",
}


def split_into_sentences(text: str) -> List[str]:
    """
    Conservative sentence split using . ? !
    This follows the previous global sentence-level script style.
    """
    s = re.sub(r"\s+", " ", text.strip())
    if not s:
        return []

    out: List[str] = []
    start = 0
    n = len(s)

    def prev_token_looks_abbrev(end_idx: int) -> bool:
        left = s[max(0, end_idx - 12): end_idx + 1].lower()
        m = re.search(r"([a-z]\.[a-z]\.|[a-z]{1,6}\.)$", left)
        if not m:
            return False
        return m.group(0) in _ABBREV

    i = 0
    while i < n:
        ch = s[i]

        if ch in ".?!":
            if prev_token_looks_abbrev(i):
                i += 1
                continue

            j = i + 1
            while j < n and s[j].isspace():
                j += 1

            if j >= n:
                sent = s[start:].strip()
                if sent:
                    out.append(sent)
                break

            nxt = s[j]
            if nxt.isupper() or nxt.isdigit() or nxt in "\"'([“":
                sent = s[start:j].strip()
                if sent:
                    out.append(sent)
                start = j
                i = j
                continue

        i += 1

    if start < n:
        tail = s[start:].strip()
        if tail and tail not in out:
            out.append(tail)

    return out if out else [s]


def word_count(x: str) -> int:
    return len([w for w in re.split(r"\s+", x.strip()) if w])


def build_global_sentence_players(units: List[str], task: TaskSpec, omit_unit_token: str):
    sentence_players: List[dict] = []
    unit_headers: List[str] = []
    unit_to_sentence_indices: List[List[int]] = []

    for u_idx, unit_text in enumerate(units):
        unit_k = u_idx + 1
        header, body = parse_unit_header(unit_text, task, unit_k)

        sentences = split_into_sentences(body)
        if not sentences:
            sentences = [omit_unit_token]

        unit_headers.append(header)
        local_sentence_indices: List[int] = []

        for s_idx, sent in enumerate(sentences):
            sent = re.sub(r"\s+", " ", sent.strip())
            if not sent:
                sent = omit_unit_token

            global_idx = len(sentence_players)

            sentence_players.append(
                {
                    "global_sentence_idx": global_idx,
                    "unit_k": unit_k,
                    "sentence_idx_in_unit": s_idx,
                    "text": sent,
                    "num_words": word_count(sent),
                    "num_chars": len(sent),
                }
            )

            local_sentence_indices.append(global_idx)

        unit_to_sentence_indices.append(local_sentence_indices)

    return sentence_players, unit_headers, unit_to_sentence_indices


# ============================================================
# SUPPORT LABELS
# ============================================================

_PUNCT = " \t\n\r.,;:!?\"'`“”‘’()[]{}"


def normalize_text(s: str) -> str:
    s = s.lower().strip(_PUNCT)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def answer_in_text(answer: str, text: str) -> bool:
    a = normalize_text(answer)
    t = normalize_text(text)
    return bool(a) and a in t


def usable_answer(answer: str, cfg: Config) -> bool:
    a = normalize_text(answer)
    if cfg.QA_EXCLUDE_YES_NO and a in {"yes", "no", "true", "false"}:
        return False
    return len(a) >= cfg.QA_MIN_ANSWER_CHARS_FOR_SUPPORT


def parse_gold_unit_ids_from_answers(answers: Sequence[str], task: TaskSpec) -> List[int]:
    ids = []
    for ans in answers:
        pat = rf"{task.unit_name}\s+(\d+)"
        for m in re.finditer(pat, ans, flags=re.IGNORECASE):
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def get_support_labels(
    *,
    task: TaskSpec,
    units: List[str],
    sentence_players: List[dict],
    answers: Sequence[str],
    cfg: Config,
) -> dict:
    """
    Returns support labels for evaluation and PR/ROC curves.

    passage_retrieval_en:
      support unit = gold paragraph from answer "Paragraph k"
      support sentence = all sentences inside gold paragraph

    hotpotqa / 2wikimqa:
      support unit = passage containing a usable gold-answer string
      support sentence = sentence containing a usable gold-answer string
    """
    if task.metric_type == "gold_paragraph":
        gold_units = parse_gold_unit_ids_from_answers(answers, task)
        support_unit_ids_1based = [k for k in gold_units if 1 <= k <= len(units)]
        support_sentence_indices = [
            sp["global_sentence_idx"]
            for sp in sentence_players
            if sp["unit_k"] in support_unit_ids_1based
        ]

        return {
            "metric_valid": bool(support_unit_ids_1based),
            "metric_reason": "gold_unit_id" if support_unit_ids_1based else "no_gold_unit_id",
            "support_unit_ids_1based": support_unit_ids_1based,
            "support_sentence_indices": support_sentence_indices,
            "usable_answers_for_matching": [],
        }

    usable_answers = [a for a in answers if usable_answer(a, cfg)]
    if not usable_answers:
        return {
            "metric_valid": False,
            "metric_reason": "no_usable_answer_for_string_match",
            "support_unit_ids_1based": [],
            "support_sentence_indices": [],
            "usable_answers_for_matching": [],
        }

    support_unit_ids_1based = []
    for i, unit in enumerate(units, start=1):
        if any(answer_in_text(a, unit) for a in usable_answers):
            support_unit_ids_1based.append(i)

    support_sentence_indices = []
    for sp in sentence_players:
        if any(answer_in_text(a, sp["text"]) for a in usable_answers):
            support_sentence_indices.append(sp["global_sentence_idx"])

    return {
        "metric_valid": bool(support_unit_ids_1based),
        "metric_reason": "answer_containment" if support_unit_ids_1based else "answer_not_found_in_context",
        "support_unit_ids_1based": support_unit_ids_1based,
        "support_sentence_indices": support_sentence_indices,
        "usable_answers_for_matching": usable_answers,
    }


# ============================================================
# PRIOR PARAGRAPH/PASSAGE TOP-K LOADING
# ============================================================

def find_prior_paragraph_shapley_csv(
    cfg: Config,
    task_name: str,
    example_index: int,
) -> Optional[Path]:
    task_dir = cfg.PARAGRAPH_OUT_ROOT / cfg.PARAGRAPH_RUN_NAME / task_name
    if not task_dir.exists():
        return None

    candidates = sorted(
        task_dir.glob(f"shard_*_of_*/examples/example_{example_index:04d}/shapley.csv")
    )
    if candidates:
        return candidates[0]
    return None


def load_prior_paragraph_topk(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {
            "prior_paragraph_shapley_csv": None,
            "paragraph_top1": None,
            "paragraph_top3": [],
            "paragraph_top5": [],
        }

    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if "player_index_1based" in row and row["player_index_1based"] != "":
                unit_k = int(row["player_index_1based"])
            elif "paragraph_k" in row and row["paragraph_k"] != "":
                unit_k = int(row["paragraph_k"])
            elif "player_idx" in row and row["player_idx"] != "":
                unit_k = int(row["player_idx"]) + 1
            else:
                continue

            phi = float(row.get("phi", row.get("raw_phi", 0.0)))
            rank = int(row.get("rank", len(rows) + 1))
            rows.append({"unit_k": unit_k, "phi": phi, "rank": rank})

    if not rows:
        return {
            "prior_paragraph_shapley_csv": str(path),
            "paragraph_top1": None,
            "paragraph_top3": [],
            "paragraph_top5": [],
        }

    rows.sort(key=lambda r: (r["rank"], -r["phi"]))

    return {
        "prior_paragraph_shapley_csv": str(path),
        "paragraph_top1": rows[0]["unit_k"],
        "paragraph_top3": [r["unit_k"] for r in rows[:3]],
        "paragraph_top5": [r["unit_k"] for r in rows[:5]],
    }


# ============================================================
# MODEL / VALUE FUNCTION
# ============================================================

def dtype_from_config(cfg: Config):
    if cfg.DTYPE == "fp16":
        return torch.float16
    if cfg.DTYPE == "fp32":
        return torch.float32
    raise ValueError(cfg.DTYPE)


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def maybe_chat(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def gpu_name_lower() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_name(0).lower()


def load_model(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID, trust_remote_code=True)

    cuda = torch.cuda.is_available()
    dtype = dtype_from_config(cfg)

    print(f"[INFO] torch.cuda.is_available()={cuda}", flush=True)
    if cuda:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Important: FlashAttention-2 does not support V100/Volta.
    # If the job lands on V100 and user requested flash_attention_2,
    # fall back to sdpa so the official run does not crash.
    requested_impl = cfg.ATTENTION_IMPL
    gpu_name = gpu_name_lower()
    if cuda and requested_impl == "flash_attention_2" and ("v100" in gpu_name or "volta" in gpu_name):
        print(
            "[WARN] Requested flash_attention_2, but current GPU looks like V100/Volta. "
            "Falling back to attn_implementation='sdpa'. Request A100/H100/Ada/Ampere nodes "
            "if you need true FlashAttention-2.",
            flush=True,
        )
        requested_impl = "sdpa"

    model_kwargs = dict(
        dtype=dtype if cuda else torch.float32,
        device_map="auto" if cuda else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation=requested_impl,
    )

    print(f"[INFO] loading model with attn_implementation={requested_impl}", flush=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_ID, **model_kwargs)
    except Exception as e:
        # Robust fallback for clusters where flash-attn is not installed or not ABI-compatible.
        if requested_impl == "flash_attention_2":
            print(
                f"[WARN] flash_attention_2 load failed: {type(e).__name__}: {e}",
                flush=True,
            )
            print("[WARN] Retrying with attn_implementation='sdpa'.", flush=True)
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_ID, **model_kwargs)
        else:
            raise

    model.eval()
    model.config.use_cache = cfg.USE_MODEL_CACHE

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] model input device={model_device(model)}", flush=True)
    print(f"[INFO] model attn_implementation={getattr(model.config, '_attn_implementation', 'unknown')}", flush=True)
    print(f"[INFO] value_batch_size={cfg.VALUE_BATCH_SIZE}", flush=True)

    return model, tokenizer

class GlobalSentenceValueFn:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        task: TaskSpec,
        sentence_players: List[dict],
        unit_headers: List[str],
        unit_to_sentence_indices: List[List[int]],
        query: str,
        gold_answer: str,
        cfg: Config,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.sentence_players = sentence_players
        self.unit_headers = unit_headers
        self.unit_to_sentence_indices = unit_to_sentence_indices
        self.query = query
        self.gold_answer = gold_answer
        self.cfg = cfg

        self.M = len(sentence_players)
        self.full_mask = (1 << self.M) - 1
        self.cache: Dict[int, float] = {}

    def subset_to_context(self, subset_mask: int) -> str:
        unit_blocks = []

        for u_idx, sent_indices in enumerate(self.unit_to_sentence_indices):
            header = self.unit_headers[u_idx]
            sentence_texts = []

            for global_idx in sent_indices:
                if (subset_mask >> global_idx) & 1:
                    sentence_texts.append(self.sentence_players[global_idx]["text"])
                else:
                    sentence_texts.append(self.cfg.OMIT_SENTENCE_TOKEN)

            body = " ".join(sentence_texts).strip()
            unit_blocks.append(f"{header} {body}".strip())

        return "\n\n".join(unit_blocks)

    def wrapped_prompt_from_subset(self, subset_mask: int) -> str:
        context = self.subset_to_context(subset_mask)
        plain_prompt = self.task.build_prompt(context, self.query)
        return maybe_chat(self.tokenizer, plain_prompt, self.cfg.USE_CHAT_TEMPLATE)

    @torch.inference_mode()
    def avg_logprob_gold_from_wrapped_prompts(self, wrapped_prompts: List[str]) -> List[float]:
        """
        Batched version of avg_logprob_gold_from_wrapped_prompt.

        This preserves the old tokenization semantics:
          prompt_ids = tokenizer(prompt, add_special_tokens=True)
          answer_ids = tokenizer(answer, add_special_tokens=False)
          input_ids = concat(prompt_ids, answer_ids)

        Then we right-pad the concatenated ids and use attention_mask.
        """
        if not wrapped_prompts:
            return []

        dev = model_device(self.model)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")

        answer_ids_cpu = self.tokenizer(
            self.gold_answer,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]

        if answer_ids_cpu.numel() == 0:
            raise ValueError("Gold answer tokenized to zero tokens.")

        T = int(answer_ids_cpu.numel())

        seqs = []
        starts = []

        for wrapped_prompt in wrapped_prompts:
            prompt_ids_cpu = self.tokenizer(
                wrapped_prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids[0]

            start = int(prompt_ids_cpu.numel())
            if start > self.cfg.MAX_PROMPT_TOKENS_WARN:
                print(
                    f"[WARN] prompt token length={start} exceeds {self.cfg.MAX_PROMPT_TOKENS_WARN}",
                    flush=True,
                )

            starts.append(start)
            seqs.append(torch.cat([prompt_ids_cpu, answer_ids_cpu], dim=0))

        max_len = max(int(x.numel()) for x in seqs)
        B = len(seqs)

        input_ids = torch.full((B, max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)

        for b, ids in enumerate(seqs):
            L = int(ids.numel())
            input_ids[b, :L] = ids
            attention_mask[b, :L] = 1

        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        answer_ids = answer_ids_cpu.to(dev)

        backbone = getattr(self.model, "model", None)
        lm_head = getattr(self.model, "lm_head", None)

        if backbone is not None and lm_head is not None:
            out = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = out.last_hidden_state

            # For each row b, answer token j is predicted by position start_b - 1 + j.
            pos = torch.tensor(
                [[starts[b] - 1 + j for j in range(T)] for b in range(B)],
                dtype=torch.long,
                device=dev,
            )
            batch_idx = torch.arange(B, device=dev).unsqueeze(1)
            hidden_ans = hidden[batch_idx, pos, :]
            logits = lm_head(hidden_ans)
        else:
            # Fallback for unusual causal LM wrappers.
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            pos = torch.tensor(
                [[starts[b] - 1 + j for j in range(T)] for b in range(B)],
                dtype=torch.long,
                device=dev,
            )
            batch_idx = torch.arange(B, device=dev).unsqueeze(1)
            logits = out.logits[batch_idx, pos, :]

        log_probs = F.log_softmax(logits.float(), dim=-1)
        gather_ids = answer_ids.view(1, T, 1).expand(B, T, 1)
        token_logprobs = log_probs.gather(-1, gather_ids).squeeze(-1)

        vals = token_logprobs.mean(dim=1).detach().cpu().tolist()

        del input_ids, attention_mask, logits, log_probs, token_logprobs
        return [float(v) for v in vals]

    @torch.inference_mode()
    def avg_logprob_gold_from_wrapped_prompt(self, wrapped_prompt: str) -> float:
        return self.avg_logprob_gold_from_wrapped_prompts([wrapped_prompt])[0]

    def f_many(self, subset_masks: Sequence[int]) -> List[float]:
        """
        Evaluate many coalition masks with cache + GPU batching.
        This is the main speedup over calling f(mask) one by one.
        """
        masks = [int(m) for m in subset_masks]
        if not masks:
            return []

        missing = []
        seen = set()
        for m in masks:
            if self.cfg.ENABLE_VALUE_CACHE and m in self.cache:
                continue
            if m in seen:
                continue
            seen.add(m)
            missing.append(m)

        if missing:
            batch_size = max(1, int(self.cfg.VALUE_BATCH_SIZE))

            for start in range(0, len(missing), batch_size):
                batch_masks = missing[start: start + batch_size]
                wrapped_prompts = [self.wrapped_prompt_from_subset(m) for m in batch_masks]
                vals = self.avg_logprob_gold_from_wrapped_prompts(wrapped_prompts)

                for m, v in zip(batch_masks, vals):
                    # Always store evaluated values in the per-shard cache.
                    # The batched DuoShap computation reads f(S) and f(complement)
                    # from this cache after pre-evaluating all required masks.
                    self.cache[int(m)] = float(v)

        return [float(self.cache[m]) for m in masks]

    def f(self, subset_mask: int) -> float:
        subset_mask = int(subset_mask)
        if self.cfg.ENABLE_VALUE_CACHE and subset_mask in self.cache:
            return self.cache[subset_mask]
        return self.f_many([subset_mask])[0]


# ============================================================
# DUOSHAP
# ============================================================

def group_shapley_value(
    value_fn: GlobalSentenceValueFn,
    subset_mask: int,
    f_empty: float,
    f_full: float,
) -> float:
    comp = value_fn.full_mask ^ subset_mask
    return 0.5 * ((f_full - value_fn.f(comp)) + (value_fn.f(subset_mask) - f_empty))


def group_shapley_value_from_cache(
    *,
    value_fn: GlobalSentenceValueFn,
    subset_mask: int,
    complement_mask: int,
    f_empty: float,
    f_full: float,
) -> float:
    return 0.5 * (
        (f_full - value_fn.cache[int(complement_mask)])
        + (value_fn.cache[int(subset_mask)] - f_empty)
    )


def sample_subset_containing_i(
    *,
    M: int,
    i: int,
    rng: np.random.Generator,
    exclude_full_set: bool,
) -> int:
    if M <= 1:
        return 1 << i

    max_k = M - 1 if exclude_full_set else M
    K = int(rng.integers(1, max_k + 1))

    mask = 1 << i

    if K > 1:
        others = np.array([j for j in range(M) if j != i], dtype=np.int64)
        chosen = rng.choice(others, size=K - 1, replace=False)
        for j in chosen.tolist():
            mask |= 1 << int(j)

    return mask


def sample_player_terms(
    *,
    player_i: int,
    value_fn: GlobalSentenceValueFn,
    cfg: Config,
    example_index: int,
) -> List[Tuple[int, int]]:
    """
    Return all (subset_mask, complement_mask) terms needed for one player.
    Sampling remains deterministic per player, preserving the original estimator.
    """
    rng = np.random.default_rng(
        stable_player_seed(cfg.SEED, cfg.TASK_NAME, example_index, player_i)
    )

    terms: List[Tuple[int, int]] = []

    for _ in range(cfg.NUM_GROUPS_PER_FEATURE):
        S = sample_subset_containing_i(
            M=value_fn.M,
            i=player_i,
            rng=rng,
            exclude_full_set=cfg.EXCLUDE_FULL_SET_IN_K,
        )

        terms.append((int(S), int(value_fn.full_mask ^ S)))

        if cfg.USE_ANTITHETIC:
            T = (value_fn.full_mask ^ S) | (1 << player_i)
            terms.append((int(T), int(value_fn.full_mask ^ T)))

    return terms


def compute_player_raw_phi(
    *,
    player_i: int,
    value_fn: GlobalSentenceValueFn,
    cfg: Config,
    f_empty: float,
    f_full: float,
    example_index: int,
) -> Tuple[float, int]:
    """
    Batched per-player version. Kept for compatibility.
    """
    terms = sample_player_terms(
        player_i=player_i,
        value_fn=value_fn,
        cfg=cfg,
        example_index=example_index,
    )

    needed_masks = []
    for S, C in terms:
        needed_masks.append(S)
        needed_masks.append(C)

    value_fn.f_many(needed_masks)

    vals = [
        group_shapley_value_from_cache(
            value_fn=value_fn,
            subset_mask=S,
            complement_mask=C,
            f_empty=f_empty,
            f_full=f_full,
        )
        for S, C in terms
    ]

    return float(np.mean(vals)), len(vals)


def compute_shard_raw_phis(
    *,
    player_indices: Sequence[int],
    value_fn: GlobalSentenceValueFn,
    cfg: Config,
    f_empty: float,
    f_full: float,
    example_index: int,
) -> Dict[int, Tuple[float, int]]:
    """
    Main speedup: pre-sample every player's terms for this sentence shard,
    evaluate every unique coalition mask in GPU batches, then compute all phis
    from the cache without more model forwards.
    """
    player_terms: Dict[int, List[Tuple[int, int]]] = {}
    needed = set()

    for player_i in player_indices:
        terms = sample_player_terms(
            player_i=int(player_i),
            value_fn=value_fn,
            cfg=cfg,
            example_index=example_index,
        )
        player_terms[int(player_i)] = terms

        for S, C in terms:
            needed.add(int(S))
            needed.add(int(C))

    needed_sorted = sorted(needed)
    print(
        f"[INFO] shard unique coalition masks to evaluate={len(needed_sorted)} "
        f"(batch_size={cfg.VALUE_BATCH_SIZE})",
        flush=True,
    )

    value_fn.f_many(needed_sorted)

    out: Dict[int, Tuple[float, int]] = {}
    for player_i, terms in player_terms.items():
        vals = [
            group_shapley_value_from_cache(
                value_fn=value_fn,
                subset_mask=S,
                complement_mask=C,
                f_empty=f_empty,
                f_full=f_full,
            )
            for S, C in terms
        ]
        out[int(player_i)] = (float(np.mean(vals)), len(vals))

    return out

# ============================================================
# ONE EXAMPLE / SHARD
# ============================================================

def build_example_state(cfg: Config, task: TaskSpec, ex: dict, example_index: int) -> dict:
    query = ex.get("input", "")
    context = ex["context"]
    answers = ex["answers"]
    gold_answer = answers[0]

    units = split_numbered_units(context, task, cfg.OMIT_UNIT_TOKEN)

    sentence_players, unit_headers, unit_to_sentence_indices = build_global_sentence_players(
        units,
        task,
        cfg.OMIT_UNIT_TOKEN,
    )

    support = get_support_labels(
        task=task,
        units=units,
        sentence_players=sentence_players,
        answers=answers,
        cfg=cfg,
    )

    prior_csv = find_prior_paragraph_shapley_csv(cfg, task.name, example_index)
    prior_topk = load_prior_paragraph_topk(prior_csv)

    return {
        "query": query,
        "answers": answers,
        "gold_answer": gold_answer,
        "units": units,
        "sentence_players": sentence_players,
        "unit_headers": unit_headers,
        "unit_to_sentence_indices": unit_to_sentence_indices,
        "support": support,
        "prior_topk": prior_topk,
    }


def run_one_sentence_shard(
    *,
    cfg: Config,
    task: TaskSpec,
    ex: dict,
    example_index: int,
    sentence_shard_id: int,
    model,
    tokenizer,
    example_dir: Path,
    array_meta: dict,
) -> dict:
    state = build_example_state(cfg, task, ex, example_index)

    sentence_players = state["sentence_players"]
    unit_headers = state["unit_headers"]
    unit_to_sentence_indices = state["unit_to_sentence_indices"]
    support = state["support"]
    prior_topk = state["prior_topk"]

    M = len(sentence_players)

    start_i = sentence_shard_id * cfg.SENTENCE_SHARD_SIZE
    end_i = min(M, start_i + cfg.SENTENCE_SHARD_SIZE)

    shard_dir = example_dir / "shards"
    shard_csv = shard_dir / f"sent_shard_{sentence_shard_id:03d}_players_{start_i}_{max(start_i, end_i) - 1}.csv"
    shard_meta = shard_dir / f"sent_shard_{sentence_shard_id:03d}_meta.json"

    structure_path = example_dir / "global_sentence_structure.json"
    write_json_once(
        structure_path,
        {
            "task": task.name,
            "example_index": example_index,
            "_id": ex.get("_id"),
            "query": state["query"],
            "answers": state["answers"],
            "gold_answer_used_for_value": state["gold_answer"],
            "metric_type": task.metric_type,
            "metric_valid": support["metric_valid"],
            "metric_reason": support["metric_reason"],
            "support_unit_ids_1based": support["support_unit_ids_1based"],
            "support_sentence_indices": support["support_sentence_indices"],
            "usable_answers_for_matching": support["usable_answers_for_matching"],
            "prior_paragraph_topk": prior_topk,
            "num_units": len(state["units"]),
            "num_sentence_players": M,
            "sentence_players": sentence_players,
            "unit_headers": unit_headers,
            "unit_to_sentence_indices": unit_to_sentence_indices,
        },
    )

    if start_i >= M:
        result = {
            "status": "no_work",
            "task": task.name,
            "example_index": example_index,
            "sentence_shard_id": sentence_shard_id,
            "start_i": start_i,
            "end_i_exclusive": end_i,
            "num_sentence_players": M,
            "reason": "sentence_shard_beyond_M",
            "array_meta": array_meta,
        }
        write_json(shard_meta, result)
        return result

    if cfg.SKIP_IF_EXISTS and shard_csv.exists() and shard_meta.exists():
        return {
            "status": "skipped_existing",
            "task": task.name,
            "example_index": example_index,
            "sentence_shard_id": sentence_shard_id,
            "start_i": start_i,
            "end_i_exclusive": end_i,
            "num_sentence_players": M,
            "shard_csv": str(shard_csv),
            "shard_meta": str(shard_meta),
            "array_meta": array_meta,
        }

    print("=" * 100, flush=True)
    print(f"[RUN] task={task.name} example={example_index} sentence_shard={sentence_shard_id}", flush=True)
    print(f"[INFO] M={M}, players [{start_i}, {end_i})", flush=True)
    print(f"[INFO] metric_valid={support['metric_valid']} reason={support['metric_reason']}", flush=True)
    print(f"[INFO] support units={support['support_unit_ids_1based']}", flush=True)
    print(f"[INFO] support sentence count={len(support['support_sentence_indices'])}", flush=True)
    print(f"[INFO] prior paragraph topk={prior_topk}", flush=True)
    print("=" * 100, flush=True)

    value_fn = GlobalSentenceValueFn(
        model=model,
        tokenizer=tokenizer,
        task=task,
        sentence_players=sentence_players,
        unit_headers=unit_headers,
        unit_to_sentence_indices=unit_to_sentence_indices,
        query=state["query"],
        gold_answer=state["gold_answer"],
        cfg=cfg,
    )

    t0 = time.time()

    print("[INFO] computing f(empty), f(full)", flush=True)
    f_empty = value_fn.f(0)
    f_full = value_fn.f(value_fn.full_mask)

    rows: List[dict] = []
    support_units = set(support["support_unit_ids_1based"])
    support_sentences = set(support["support_sentence_indices"])

    top1 = prior_topk.get("paragraph_top1")
    top3 = set(prior_topk.get("paragraph_top3", []))
    top5 = set(prior_topk.get("paragraph_top5", []))

    shard_phi = compute_shard_raw_phis(
        player_indices=list(range(start_i, end_i)),
        value_fn=value_fn,
        cfg=cfg,
        f_empty=f_empty,
        f_full=f_full,
        example_index=example_index,
    )

    for local_pos, i in enumerate(range(start_i, end_i), start=1):
        raw_phi, count = shard_phi[int(i)]

        sp = sentence_players[i]
        unit_k = int(sp["unit_k"])

        rows.append(
            {
                "task": task.name,
                "example_index": example_index,
                "global_sentence_idx": i,
                "unit_k": unit_k,
                "sentence_idx_in_unit": sp["sentence_idx_in_unit"],
                "raw_phi": raw_phi,
                "count": count,
                "is_support_unit": int(unit_k in support_units),
                "is_support_sentence": int(i in support_sentences),
                "metric_valid": int(bool(support["metric_valid"])),
                "is_in_paragraph_level_top1": int(top1 is not None and unit_k == top1),
                "is_in_paragraph_level_top3": int(unit_k in top3),
                "is_in_paragraph_level_top5": int(unit_k in top5),
                "num_words": sp["num_words"],
                "num_chars": sp["num_chars"],
                "text": sp["text"],
            }
        )

        print(
            f"[PROGRESS] example={example_index} shard={sentence_shard_id} "
            f"player={i} local={local_pos}/{end_i - start_i} "
            f"raw_phi={raw_phi:.6f} cache={len(value_fn.cache)}",
            flush=True,
        )

    write_csv(
        shard_csv,
        rows,
        fieldnames=[
            "task",
            "example_index",
            "global_sentence_idx",
            "unit_k",
            "sentence_idx_in_unit",
            "raw_phi",
            "count",
            "is_support_unit",
            "is_support_sentence",
            "metric_valid",
            "is_in_paragraph_level_top1",
            "is_in_paragraph_level_top3",
            "is_in_paragraph_level_top5",
            "num_words",
            "num_chars",
            "text",
        ],
    )

    elapsed = time.time() - t0

    if cfg.SAVE_VALUE_CACHE:
        write_json(
            shard_dir / f"sent_shard_{sentence_shard_id:03d}_value_cache.json",
            {"subset_mask_to_value": {str(k): float(v) for k, v in value_fn.cache.items()}},
        )

    result = {
        "status": "ok",
        "task": task.name,
        "example_index": example_index,
        "sentence_shard_id": sentence_shard_id,
        "start_i": start_i,
        "end_i_exclusive": end_i,
        "num_players_in_shard": end_i - start_i,
        "num_sentence_players": M,
        "num_units": len(state["units"]),
        "metric_valid": support["metric_valid"],
        "metric_reason": support["metric_reason"],
        "support_unit_ids_1based": support["support_unit_ids_1based"],
        "num_support_sentences": len(support["support_sentence_indices"]),
        "f_empty": float(f_empty),
        "f_full": float(f_full),
        "f_full_minus_empty": float(f_full - f_empty),
        "raw_phi_sum_shard": float(sum(r["raw_phi"] for r in rows)),
        "elapsed_seconds": float(elapsed),
        "value_cache_size": len(value_fn.cache),
        "shard_csv": str(shard_csv),
        "shard_meta": str(shard_meta),
        "structure_path": str(structure_path),
        "num_groups_per_feature": cfg.NUM_GROUPS_PER_FEATURE,
        "value_batch_size": cfg.VALUE_BATCH_SIZE,
        "attention_impl_requested": cfg.ATTENTION_IMPL,
        "use_antithetic": cfg.USE_ANTITHETIC,
        "exclude_full_set_in_k": cfg.EXCLUDE_FULL_SET_IN_K,
        "array_meta": array_meta,
    }

    write_json(shard_meta, result)

    del value_fn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[DONE] example={example_index} sent_shard={sentence_shard_id} "
        f"elapsed={elapsed / 60:.2f} min",
        flush=True,
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    cfg = parse_args()
    configure_environment(cfg)

    if cfg.TASK_NAME not in TASKS:
        raise ValueError(f"Unsupported task: {cfg.TASK_NAME}")

    task = TASKS[cfg.TASK_NAME]

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    data_path = cfg.LONGBENCH_DATA_DIR / f"{cfg.TASK_NAME}.jsonl"
    examples = load_jsonl(data_path)

    selected = select_indices(len(examples), cfg)

    example_index, sentence_shard_ids, array_meta = map_array_id_to_work(cfg, selected)

    run_root = cfg.OUT_DIR / cfg.RUN_NAME / cfg.TASK_NAME
    run_root.mkdir(parents=True, exist_ok=True)

    write_json_once(
        run_root / "run_config.json",
        {
            "config": asdict(cfg),
            "task": cfg.TASK_NAME,
            "num_examples_in_file": len(examples),
            "selected_indices": selected,
            "selected_indices_json": (
                str(cfg.SELECTED_INDICES_JSON)
                if cfg.SELECTED_INDICES_JSON is not None
                else None
            ),
            "note": "Array mapping is per job; see job_logs/array_*.json for exact array mappings.",
        },
    )

    print("=" * 100, flush=True)
    print("[INFO] Global sentence-level DuoShap LongBench worker", flush=True)
    print(f"[TASK] {cfg.TASK_NAME}", flush=True)
    print(f"[DATA] {data_path}", flush=True)
    print(f"[OUT]  {run_root}", flush=True)
    print(f"[SELECTED] {selected}", flush=True)
    print(f"[ARRAY] array_id={cfg.ARRAY_ID}", flush=True)
    print(f"[ARRAY] mapping={array_meta}", flush=True)
    print(f"[INFO] selected examples={len(selected)}", flush=True)
    print("=" * 100, flush=True)

    if example_index is None or sentence_shard_ids is None:
        print("[NO WORK] array id maps beyond selected examples.", flush=True)
        return

    ex = examples[example_index]
    example_dir = run_root / f"example_{example_index:04d}"

    model, tokenizer = load_model(cfg)

    job_results = []

    for sentence_shard_id in sentence_shard_ids:
        try:
            result = run_one_sentence_shard(
                cfg=cfg,
                task=task,
                ex=ex,
                example_index=example_index,
                sentence_shard_id=sentence_shard_id,
                model=model,
                tokenizer=tokenizer,
                example_dir=example_dir,
                array_meta=array_meta,
            )
            job_results.append(result)

        except Exception as e:
            error_dir = example_dir / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            err = {
                "status": "error",
                "task": task.name,
                "example_index": example_index,
                "sentence_shard_id": sentence_shard_id,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "array_meta": array_meta,
            }
            write_json(error_dir / f"sent_shard_{sentence_shard_id:03d}_error.json", err)
            print(
                f"[ERROR] example={example_index} sent_shard={sentence_shard_id}: {repr(e)}",
                flush=True,
            )
            print(traceback.format_exc(), flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    job_log_dir = run_root / "job_logs"
    job_log_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        job_log_dir / f"array_{cfg.ARRAY_ID:06d}.json",
        {
            "array_id": cfg.ARRAY_ID,
            "task": task.name,
            "example_index": example_index,
            "sentence_shard_ids": sentence_shard_ids,
            "results": job_results,
        },
    )

    print("[DONE] array worker finished.", flush=True)


if __name__ == "__main__":
    main()