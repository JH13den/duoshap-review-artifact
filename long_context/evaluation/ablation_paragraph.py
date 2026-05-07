#!/usr/bin/env python3
"""
Dataset-level paragraph/passage ablation for LongBench DuoShap results.

Supported tasks:
  - passage_retrieval_en
  - hotpotqa
  - 2wikimqa

Goal:
  For each example with paragraph/passage-level DuoShap scores:
    1. Load paragraph/passage players.
    2. Load DuoShap phi from previous paragraph-level result.
    3. Start from full context.
    4. Remove high-DuoShap players first.
    5. Remove low-DuoShap players first.
    6. Optionally remove random players as a baseline.
    7. Evaluate f(S) = avg log p(gold answer | masked prompt).
    8. Save raw and normalized curves.

Output layout:
  output/ablation_longbench/<ABLATION_RUN_NAME>/<TASK>/shard_000_of_004/
    config.json
    results.jsonl
    errors.jsonl
    summary.json
    examples/example_XXXX/
      ablation_curves.csv
      random_repeats.csv
      ablation_result.json
      ranking.csv

Notes:
  - This script does not recompute DuoShap.
  - It only uses existing paragraph/passage-level shapley.csv files.
  - Slurm controls GPU allocation.
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
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    HF_HOME: Path = Path("xxxxxx")

    LONGBENCH_DATA_DIR: Path = Path(
        "xxxxxx"
    )

    # Previous paragraph-level DuoShap outputs.
    DUOSHAP_OUT_ROOT: Path = Path(
        "xxxxxx"
    )
    DUOSHAP_RUN_NAME: str = "duoshap_qwen25_7b_paragraph_level"

    # New ablation outputs.
    ABLATION_OUT_ROOT: Path = Path(
        "xxxxxx"
    )
    ABLATION_RUN_NAME: str = "paragraph_ablation_qwen25_7b"

    MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"
    DTYPE: str = "fp16"
    USE_CHAT_TEMPLATE: bool = True
    USE_MODEL_CACHE: bool = False

    TASK_NAME: str = "passage_retrieval_en"

    EXAMPLE_START: int = 0
    EXAMPLE_END: Optional[int] = None
    MAX_EXAMPLES: Optional[int] = None

    NUM_SHARDS: int = 1
    SHARD_ID: int = 0

    SEED: int = 42

    OMIT_TOKEN: str = "[OMITTED]"
    ENABLE_VALUE_CACHE: bool = True
    MAX_PROMPT_TOKENS_WARN: int = 40000

    # Main ablation curve settings.
    # Percent grid is better than raw k because task player counts differ.
    PERCENT_GRID: str = "0,10,20,30,40,50,60,70,80,90,100"

    # Random removal baseline.
    RANDOM_REPEATS: int = 5

    # QA answer-containment metric settings.
    QA_MIN_ANSWER_CHARS_FOR_SUPPORT: int = 3
    QA_EXCLUDE_YES_NO: bool = True

    # Normalize as (f(S)-f_empty)/(f_full-f_empty).
    # If f_full - f_empty is too small, normalized value is written as NaN.
    NORMALIZATION_EPS: float = 1e-8

    # Run all examples, but save flags so later plotting can filter:
    # metric_valid=True and normalization_valid=True.
    SKIP_INVALID_METRIC: bool = False

    RESUME_SKIP_COMPLETED: bool = True


# ============================================================
# ARGUMENTS
# ============================================================

def parse_optional_int(x: str) -> Optional[int]:
    if x is None:
        return None
    if str(x).lower() in {"none", "null", "-1"}:
        return None
    return int(x)


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser()

    p.add_argument("--task", type=str, default=cfg.TASK_NAME,
                   choices=["passage_retrieval_en", "hotpotqa", "2wikimqa"])

    p.add_argument("--duoshap-out-root", type=Path, default=cfg.DUOSHAP_OUT_ROOT)
    p.add_argument("--duoshap-run-name", type=str, default=cfg.DUOSHAP_RUN_NAME)

    p.add_argument("--ablation-out-root", type=Path, default=cfg.ABLATION_OUT_ROOT)
    p.add_argument("--ablation-run-name", type=str, default=cfg.ABLATION_RUN_NAME)

    p.add_argument("--hf-home", type=Path, default=cfg.HF_HOME)
    p.add_argument("--data-dir", type=Path, default=cfg.LONGBENCH_DATA_DIR)

    p.add_argument("--model-id", type=str, default=cfg.MODEL_ID)
    p.add_argument("--dtype", type=str, default=cfg.DTYPE, choices=["fp16", "fp32"])

    p.add_argument("--example-start", type=int, default=cfg.EXAMPLE_START)
    p.add_argument("--example-end", type=parse_optional_int, default=cfg.EXAMPLE_END)
    p.add_argument("--max-examples", type=parse_optional_int, default=cfg.MAX_EXAMPLES)

    p.add_argument("--num-shards", type=int, default=cfg.NUM_SHARDS)
    p.add_argument("--shard-id", type=int, default=cfg.SHARD_ID)

    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--percent-grid", type=str, default=cfg.PERCENT_GRID)
    p.add_argument("--random-repeats", type=int, default=cfg.RANDOM_REPEATS)

    p.add_argument("--skip-invalid-metric", action="store_true")
    p.add_argument("--no-value-cache", action="store_true")
    p.add_argument("--no-resume", action="store_true")

    args = p.parse_args()

    cfg.TASK_NAME = args.task

    cfg.DUOSHAP_OUT_ROOT = args.duoshap_out_root
    cfg.DUOSHAP_RUN_NAME = args.duoshap_run_name

    cfg.ABLATION_OUT_ROOT = args.ablation_out_root
    cfg.ABLATION_RUN_NAME = args.ablation_run_name

    cfg.HF_HOME = args.hf_home
    cfg.LONGBENCH_DATA_DIR = args.data_dir

    cfg.MODEL_ID = args.model_id
    cfg.DTYPE = args.dtype

    cfg.EXAMPLE_START = args.example_start
    cfg.EXAMPLE_END = args.example_end
    cfg.MAX_EXAMPLES = args.max_examples

    cfg.NUM_SHARDS = args.num_shards
    cfg.SHARD_ID = args.shard_id

    cfg.SEED = args.seed
    cfg.PERCENT_GRID = args.percent_grid
    cfg.RANDOM_REPEATS = args.random_repeats

    cfg.SKIP_INVALID_METRIC = args.skip_invalid_metric
    cfg.ENABLE_VALUE_CACHE = not args.no_value_cache
    cfg.RESUME_SKIP_COMPLETED = not args.no_resume

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
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n")
        f.flush()


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def select_indices(n: int, cfg: Config) -> List[int]:
    start = max(0, cfg.EXAMPLE_START)
    end = cfg.EXAMPLE_END if cfg.EXAMPLE_END is not None else n
    end = min(end, n)

    indices = list(range(start, end))
    if cfg.MAX_EXAMPLES is not None:
        indices = indices[: cfg.MAX_EXAMPLES]
    return indices


def shard_indices(indices: Sequence[int], shard_id: int, num_shards: int) -> List[int]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Bad shard_id={shard_id}, num_shards={num_shards}")

    n = len(indices)
    start = math.floor(n * shard_id / num_shards)
    end = math.floor(n * (shard_id + 1) / num_shards)
    return list(indices[start:end])


def completed_indices(results_jsonl: Path) -> Set[int]:
    done = set()
    if not results_jsonl.exists():
        return done

    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    done.add(int(row["example_index"]))
            except Exception:
                pass
    return done


def stable_seed(base_seed: int, task_name: str, example_index: int, offset: int = 0) -> int:
    h = sum((i + 1) * ord(c) for i, c in enumerate(task_name))
    return int((base_seed + 1009 * example_index + 9173 * h + 10007 * offset) % (2**32 - 1))


def parse_percent_grid(s: str) -> List[float]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    vals = sorted(set(vals))
    if 0.0 not in vals:
        vals = [0.0] + vals
    if 100.0 not in vals:
        vals = vals + [100.0]
    return vals


def nan_if_none(x):
    return float("nan") if x is None else x


# ============================================================
# TASK SPECS
# ============================================================

@dataclass(frozen=True)
class TaskSpec:
    name: str
    unit_name: str
    metric_type: str
    build_prompt: Callable[[str, str], str]
    split_players: Callable[[str], List[str]]


_PAR_HDR = re.compile(r"(Paragraph\s+(\d+)\s*:)", flags=re.IGNORECASE)
_PASSAGE_HDR = re.compile(r"(Passage\s+(\d+)\s*:)", flags=re.IGNORECASE)


def split_numbered_units(
    context: str,
    header_regex: re.Pattern,
    unit_name: str,
    expected_n: Optional[int],
    omit_token: str,
) -> List[str]:
    matches = list(header_regex.finditer(context))

    if matches:
        chunks: Dict[int, str] = {}

        for i, m in enumerate(matches):
            k = int(m.group(2))
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
            chunks[k] = context[start:end].strip()

        max_k = expected_n if expected_n is not None else max(chunks.keys())

        players = []
        for k in range(1, max_k + 1):
            if k in chunks:
                players.append(chunks[k])
            else:
                players.append(f"{unit_name} {k}: {omit_token}")
        return players

    parts = [p.strip() for p in re.split(r"\n\s*\n", context) if p.strip()]
    if expected_n is not None:
        parts = parts[:expected_n]

    players = [f"{unit_name} {i + 1}: {p}" for i, p in enumerate(parts)]

    if expected_n is not None:
        while len(players) < expected_n:
            k = len(players) + 1
            players.append(f"{unit_name} {k}: {omit_token}")

    return players


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


def split_retrieval_players(context: str) -> List[str]:
    return split_numbered_units(
        context=context,
        header_regex=_PAR_HDR,
        unit_name="Paragraph",
        expected_n=30,
        omit_token="[OMITTED]",
    )


def split_qa_players(context: str) -> List[str]:
    return split_numbered_units(
        context=context,
        header_regex=_PASSAGE_HDR,
        unit_name="Passage",
        expected_n=None,
        omit_token="[OMITTED]",
    )


TASKS: Dict[str, TaskSpec] = {
    "passage_retrieval_en": TaskSpec(
        name="passage_retrieval_en",
        unit_name="Paragraph",
        metric_type="gold_paragraph",
        build_prompt=build_prompt_passage_retrieval,
        split_players=split_retrieval_players,
    ),
    "hotpotqa": TaskSpec(
        name="hotpotqa",
        unit_name="Passage",
        metric_type="answer_containment",
        build_prompt=build_prompt_qa,
        split_players=split_qa_players,
    ),
    "2wikimqa": TaskSpec(
        name="2wikimqa",
        unit_name="Passage",
        metric_type="answer_containment",
        build_prompt=build_prompt_qa,
        split_players=split_qa_players,
    ),
}


# ============================================================
# METRIC HELPERS
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


def parse_gold_paragraph_ids(answers: Sequence[str]) -> List[int]:
    ids = []
    for a in answers:
        for m in re.finditer(r"Paragraph\s+(\d+)", a, flags=re.IGNORECASE):
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def get_support_ids(
    task: TaskSpec,
    players: List[str],
    answers: Sequence[str],
    cfg: Config,
) -> Tuple[List[int], bool, str]:
    if task.metric_type == "gold_paragraph":
        gold_ids = parse_gold_paragraph_ids(answers)
        support = [k - 1 for k in gold_ids if 1 <= k <= len(players)]
        if not support:
            return [], False, "no_gold_paragraph_id"
        return support, True, "gold_paragraph_id"

    usable_answers = [a for a in answers if usable_answer(a, cfg)]
    if not usable_answers:
        return [], False, "no_usable_answer_for_string_match"

    support = []
    for i, p in enumerate(players):
        if any(answer_in_text(a, p) for a in usable_answers):
            support.append(i)

    if not support:
        return [], False, "answer_not_found_in_context"

    return support, True, "answer_containment"


# ============================================================
# MODEL + VALUE FUNCTION
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


def load_model(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID, trust_remote_code=True)

    cuda = torch.cuda.is_available()
    dtype = dtype_from_config(cfg)

    print(f"[INFO] torch.cuda.is_available()={cuda}", flush=True)
    if cuda:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_ID,
        torch_dtype=dtype if cuda else torch.float32,
        device_map="auto" if cuda else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.eval()
    model.config.use_cache = cfg.USE_MODEL_CACHE

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] model input device={model_device(model)}", flush=True)

    return model, tokenizer


class LongContextValueFn:
    def __init__(
        self,
        model,
        tokenizer,
        task: TaskSpec,
        players: List[str],
        query: str,
        gold_answer: str,
        cfg: Config,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.players = players
        self.query = query
        self.gold_answer = gold_answer
        self.cfg = cfg

        self.M = len(players)
        self.full_mask = (1 << self.M) - 1
        self.cache: Dict[int, float] = {}

    def subset_to_context(self, mask: int) -> str:
        blocks = []

        for i, original in enumerate(self.players):
            if (mask >> i) & 1:
                blocks.append(original)
            else:
                blocks.append(f"{self.task.unit_name} {i + 1}: {self.cfg.OMIT_TOKEN}")

        return "\n\n".join(blocks)

    @torch.inference_mode()
    def avg_logprob_gold_from_wrapped_prompt(self, wrapped_prompt: str) -> float:
        dev = model_device(self.model)

        prompt_ids = self.tokenizer(
            wrapped_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(dev)

        answer_ids = self.tokenizer(
            self.gold_answer,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(dev)

        if answer_ids.shape[1] == 0:
            raise ValueError("Gold answer tokenized to zero tokens.")

        start = prompt_ids.shape[1]
        T = answer_ids.shape[1]

        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)

        backbone = getattr(self.model, "model", None)
        lm_head = getattr(self.model, "lm_head", None)

        if backbone is not None and lm_head is not None:
            out = backbone(input_ids=input_ids, use_cache=False, return_dict=True)
            hidden = out.last_hidden_state
            logits = lm_head(hidden[:, start - 1 : start - 1 + T, :])
        else:
            out = self.model(input_ids=input_ids, use_cache=False, return_dict=True)
            logits = out.logits[:, start - 1 : start - 1 + T, :]

        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)

        return float(token_logprobs.mean().item())

    def f(self, mask: int) -> float:
        if self.cfg.ENABLE_VALUE_CACHE and mask in self.cache:
            return self.cache[mask]

        context = self.subset_to_context(mask)
        plain_prompt = self.task.build_prompt(context, self.query)
        wrapped_prompt = maybe_chat(self.tokenizer, plain_prompt, self.cfg.USE_CHAT_TEMPLATE)

        token_len = self.tokenizer(
            wrapped_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.shape[1]

        if token_len > self.cfg.MAX_PROMPT_TOKENS_WARN:
            print(
                f"[WARN] prompt token length={token_len} exceeds {self.cfg.MAX_PROMPT_TOKENS_WARN}",
                flush=True,
            )

        value = self.avg_logprob_gold_from_wrapped_prompt(wrapped_prompt)

        if self.cfg.ENABLE_VALUE_CACHE:
            self.cache[mask] = value

        return value


# ============================================================
# SHAPLEY LOADING
# ============================================================

def find_shapley_csv(cfg: Config, task_name: str, example_index: int) -> Optional[Path]:
    task_dir = cfg.DUOSHAP_OUT_ROOT / cfg.DUOSHAP_RUN_NAME / task_name
    if not task_dir.exists():
        return None

    candidates = sorted(
        task_dir.glob(f"shard_*_of_*/examples/example_{example_index:04d}/shapley.csv")
    )
    if candidates:
        return candidates[0]
    return None


def read_duoshap_shapley_csv(path: Path, M: int) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Expected current paragraph-level shapley.csv fields:
      rank, player_index_1based, phi, count, is_support, text_preview

    Returns:
      phi: length M, indexed 0-based
      counts: length M
      ranking_rows: sorted by rank
    """
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing shapley.csv: {path}")

    phi = np.full(M, np.nan, dtype=np.float64)
    counts = np.zeros(M, dtype=np.int64)
    ranking_rows = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "player_index_1based" not in row:
                raise ValueError(f"{path} missing player_index_1based column")

            idx1 = int(row["player_index_1based"])
            idx0 = idx1 - 1
            if idx0 < 0 or idx0 >= M:
                continue

            val = float(row["phi"])
            cnt = int(float(row.get("count", "0")))

            phi[idx0] = val
            counts[idx0] = cnt

            ranking_rows.append(
                {
                    "rank": int(float(row.get("rank", len(ranking_rows) + 1))),
                    "player_index_1based": idx1,
                    "phi": val,
                    "count": cnt,
                    "is_support": str(row.get("is_support", "")).lower() in {"true", "1", "yes"},
                    "text_preview": row.get("text_preview", ""),
                }
            )

    missing = [i + 1 for i, v in enumerate(phi) if np.isnan(v)]
    if missing:
        raise ValueError(f"Missing phi values for players {missing} in {path}")

    ranking_rows.sort(key=lambda r: r["rank"])

    return phi, counts, ranking_rows


# ============================================================
# ABLATION HELPERS
# ============================================================

def remove_players_from_full_mask(full_mask: int, removal_order_zero_based: Sequence[int], k: int) -> int:
    keep_mask = full_mask
    for idx0 in removal_order_zero_based[:k]:
        keep_mask &= ~(1 << int(idx0))
    return keep_mask


def normalize_value(f_value: float, f_empty: float, f_full: float, eps: float) -> Optional[float]:
    denom = f_full - f_empty
    if abs(denom) <= eps:
        return None
    return (f_value - f_empty) / denom


def compute_ablation_curves(
    *,
    value_fn: LongContextValueFn,
    phi: np.ndarray,
    cfg: Config,
    task_name: str,
    example_index: int,
    percent_grid: List[float],
) -> Tuple[List[dict], List[dict], dict]:
    M = value_fn.M
    full_mask = value_fn.full_mask

    order_desc = np.argsort(-phi).astype(int).tolist()
    order_asc = np.argsort(phi).astype(int).tolist()

    rng = np.random.default_rng(stable_seed(cfg.SEED, task_name, example_index, offset=777))
    random_orders = []
    for r in range(cfg.RANDOM_REPEATS):
        random_orders.append(rng.permutation(np.arange(M)).astype(int).tolist())

    f_empty = value_fn.f(0)
    f_full = value_fn.f(full_mask)
    normalization_valid = abs(f_full - f_empty) > cfg.NORMALIZATION_EPS

    curve_rows = []
    random_repeat_rows = []

    for p in percent_grid:
        k = int(round((p / 100.0) * M))
        k = max(0, min(M, k))

        top_mask = remove_players_from_full_mask(full_mask, order_desc, k)
        bottom_mask = remove_players_from_full_mask(full_mask, order_asc, k)

        f_top = value_fn.f(top_mask)
        f_bottom = value_fn.f(bottom_mask)

        n_top = normalize_value(f_top, f_empty, f_full, cfg.NORMALIZATION_EPS)
        n_bottom = normalize_value(f_bottom, f_empty, f_full, cfg.NORMALIZATION_EPS)

        random_vals = []
        random_norms = []

        for r, order in enumerate(random_orders):
            rand_mask = remove_players_from_full_mask(full_mask, order, k)
            f_rand = value_fn.f(rand_mask)
            n_rand = normalize_value(f_rand, f_empty, f_full, cfg.NORMALIZATION_EPS)

            random_vals.append(f_rand)
            if n_rand is not None:
                random_norms.append(n_rand)

            random_repeat_rows.append(
                {
                    "task": task_name,
                    "example_index": example_index,
                    "percent_removed": p,
                    "k_removed": k,
                    "repeat": r,
                    "keep_mask": rand_mask,
                    "f_random": f_rand,
                    "normalized_random": nan_if_none(n_rand),
                    "removed_random_1based": json.dumps([int(i + 1) for i in order[:k]]),
                }
            )

        f_random_mean = float(np.mean(random_vals)) if random_vals else float("nan")
        f_random_std = float(np.std(random_vals)) if random_vals else float("nan")

        if random_norms:
            n_random_mean = float(np.mean(random_norms))
            n_random_std = float(np.std(random_norms))
        else:
            n_random_mean = float("nan")
            n_random_std = float("nan")

        curve_rows.append(
            {
                "task": task_name,
                "example_index": example_index,
                "percent_removed": p,
                "k_removed": k,
                "num_players": M,

                "f_empty": f_empty,
                "f_full": f_full,
                "f_full_minus_empty": f_full - f_empty,
                "normalization_valid": normalization_valid,

                "top_keep_mask": top_mask,
                "bottom_keep_mask": bottom_mask,

                "f_remove_top": f_top,
                "f_remove_bottom": f_bottom,
                "f_remove_random_mean": f_random_mean,
                "f_remove_random_std": f_random_std,

                "delta_remove_top": f_top - f_full,
                "delta_remove_bottom": f_bottom - f_full,
                "delta_remove_random_mean": f_random_mean - f_full,

                "normalized_remove_top": nan_if_none(n_top),
                "normalized_remove_bottom": nan_if_none(n_bottom),
                "normalized_remove_random_mean": n_random_mean,
                "normalized_remove_random_std": n_random_std,

                "removed_top_1based": json.dumps([int(i + 1) for i in order_desc[:k]]),
                "removed_bottom_1based": json.dumps([int(i + 1) for i in order_asc[:k]]),
            }
        )

    debug = {
        "num_players": M,
        "f_empty": f_empty,
        "f_full": f_full,
        "f_full_minus_empty": f_full - f_empty,
        "normalization_valid": normalization_valid,
        "order_desc_1based": [int(i + 1) for i in order_desc],
        "order_asc_1based": [int(i + 1) for i in order_asc],
        "random_orders_1based": [[int(i + 1) for i in order] for order in random_orders],
        "num_cached_subsets": len(value_fn.cache),
    }

    return curve_rows, random_repeat_rows, debug


# ============================================================
# OUTPUT SUMMARY
# ============================================================

def summarize_results(rows: List[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    skipped = [r for r in rows if r.get("status") == "skipped_invalid_metric"]

    runtimes = [float(r["runtime_sec"]) for r in ok if r.get("runtime_sec") is not None]
    metric_valid = [r for r in ok if r.get("metric_valid") is True]
    norm_valid = [r for r in ok if r.get("normalization_valid") is True]

    return {
        "num_ok": len(ok),
        "num_skipped_invalid_metric": len(skipped),
        "num_metric_valid": len(metric_valid),
        "num_normalization_valid": len(norm_valid),
        "avg_runtime_sec": float(np.mean(runtimes)) if runtimes else None,
        "std_runtime_sec": float(np.std(runtimes)) if runtimes else None,
    }


# ============================================================
# ONE EXAMPLE
# ============================================================

def run_one_example(
    *,
    task: TaskSpec,
    ex: dict,
    example_index: int,
    model,
    tokenizer,
    cfg: Config,
    example_dir: Path,
    percent_grid: List[float],
) -> dict:
    t0 = time.time()

    query = ex["input"]
    context = ex["context"]
    answers = ex["answers"]
    gold_answer = answers[0]

    players = task.split_players(context)
    M = len(players)

    support_ids, metric_valid, metric_reason = get_support_ids(
        task=task,
        players=players,
        answers=answers,
        cfg=cfg,
    )

    if cfg.SKIP_INVALID_METRIC and not metric_valid:
        result = {
            "status": "skipped_invalid_metric",
            "task": task.name,
            "example_index": example_index,
            "metric_valid": metric_valid,
            "metric_reason": metric_reason,
            "num_players": M,
        }
        write_json(example_dir / "ablation_result.json", result)
        return result

    shapley_csv = find_shapley_csv(cfg, task.name, example_index)
    if shapley_csv is None:
        raise FileNotFoundError(
            f"Cannot find shapley.csv for task={task.name}, example={example_index}"
        )

    phi, counts, ranking_rows = read_duoshap_shapley_csv(shapley_csv, M)

    value_fn = LongContextValueFn(
        model=model,
        tokenizer=tokenizer,
        task=task,
        players=players,
        query=query,
        gold_answer=gold_answer,
        cfg=cfg,
    )

    curve_rows, random_repeat_rows, debug = compute_ablation_curves(
        value_fn=value_fn,
        phi=phi,
        cfg=cfg,
        task_name=task.name,
        example_index=example_index,
        percent_grid=percent_grid,
    )

    example_dir.mkdir(parents=True, exist_ok=True)

    curve_fields = [
        "task",
        "example_index",
        "percent_removed",
        "k_removed",
        "num_players",
        "f_empty",
        "f_full",
        "f_full_minus_empty",
        "normalization_valid",
        "top_keep_mask",
        "bottom_keep_mask",
        "f_remove_top",
        "f_remove_bottom",
        "f_remove_random_mean",
        "f_remove_random_std",
        "delta_remove_top",
        "delta_remove_bottom",
        "delta_remove_random_mean",
        "normalized_remove_top",
        "normalized_remove_bottom",
        "normalized_remove_random_mean",
        "normalized_remove_random_std",
        "removed_top_1based",
        "removed_bottom_1based",
    ]

    write_csv(example_dir / "ablation_curves.csv", curve_rows, curve_fields)

    random_fields = [
        "task",
        "example_index",
        "percent_removed",
        "k_removed",
        "repeat",
        "keep_mask",
        "f_random",
        "normalized_random",
        "removed_random_1based",
    ]

    write_csv(example_dir / "random_repeats.csv", random_repeat_rows, random_fields)

    ranking_fields = [
        "rank",
        "player_index_1based",
        "phi",
        "count",
        "is_support",
        "text_preview",
    ]
    write_csv(example_dir / "ranking.csv", ranking_rows, ranking_fields)

    top_idx = int(np.argmax(phi))
    bottom_idx = int(np.argmin(phi))

    result = {
        "status": "ok",
        "task": task.name,
        "example_index": example_index,
        "_id": ex.get("_id"),
        "metric_type": task.metric_type,
        "metric_valid": metric_valid,
        "metric_reason": metric_reason,
        "answers": answers,
        "gold_answer_used_for_value": gold_answer,
        "num_players": M,
        "support_ids_1based": [i + 1 for i in support_ids],
        "top_player_1based": top_idx + 1,
        "bottom_player_1based": bottom_idx + 1,
        "top_score": float(phi[top_idx]),
        "bottom_score": float(phi[bottom_idx]),
        "top1_hit": bool(metric_valid and top_idx in set(support_ids)),
        "f_empty": float(debug["f_empty"]),
        "f_full": float(debug["f_full"]),
        "f_full_minus_empty": float(debug["f_full_minus_empty"]),
        "normalization_valid": bool(debug["normalization_valid"]),
        "num_curve_points": len(curve_rows),
        "random_repeats": cfg.RANDOM_REPEATS,
        "percent_grid": percent_grid,
        "runtime_sec": float(time.time() - t0),
        "value_cache_size": int(debug["num_cached_subsets"]),
        "shapley_csv": str(shapley_csv),
        "ablation_curves_csv": str(example_dir / "ablation_curves.csv"),
        "random_repeats_csv": str(example_dir / "random_repeats.csv"),
        "ranking_csv": str(example_dir / "ranking.csv"),
        "debug": debug,
    }

    write_json(example_dir / "ablation_result.json", result)

    del value_fn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    cfg = parse_args()
    configure_environment(cfg)

    if cfg.TASK_NAME not in TASKS:
        raise ValueError(f"Unknown task: {cfg.TASK_NAME}")

    task = TASKS[cfg.TASK_NAME]
    percent_grid = parse_percent_grid(cfg.PERCENT_GRID)

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    data_path = cfg.LONGBENCH_DATA_DIR / f"{cfg.TASK_NAME}.jsonl"
    examples = load_jsonl(data_path)

    selected = select_indices(len(examples), cfg)
    selected_shard = shard_indices(selected, cfg.SHARD_ID, cfg.NUM_SHARDS)

    run_root = (
        cfg.ABLATION_OUT_ROOT
        / cfg.ABLATION_RUN_NAME
        / cfg.TASK_NAME
        / f"shard_{cfg.SHARD_ID:03d}_of_{cfg.NUM_SHARDS:03d}"
    )
    examples_root = run_root / "examples"

    run_root.mkdir(parents=True, exist_ok=True)

    write_json(
        run_root / "config.json",
        {
            "config": asdict(cfg),
            "num_examples_in_file": len(examples),
            "selected_indices_before_shard": selected,
            "selected_indices_this_shard": selected_shard,
            "percent_grid": percent_grid,
        },
    )

    results_jsonl = run_root / "results.jsonl"
    errors_jsonl = run_root / "errors.jsonl"

    done = completed_indices(results_jsonl) if cfg.RESUME_SKIP_COMPLETED else set()

    print("=" * 100, flush=True)
    print("[INFO] Paragraph/passage-level ablation worker", flush=True)
    print(f"[TASK] {cfg.TASK_NAME}", flush=True)
    print(f"[DATA] {data_path}", flush=True)
    print(f"[OUT]  {run_root}", flush=True)
    print(f"[INFO] examples in file: {len(examples)}", flush=True)
    print(f"[INFO] selected before shard: {len(selected)}", flush=True)
    print(f"[INFO] shard {cfg.SHARD_ID}/{cfg.NUM_SHARDS}: {len(selected_shard)} examples", flush=True)
    print(f"[INFO] percent_grid={percent_grid}", flush=True)
    print(f"[INFO] random_repeats={cfg.RANDOM_REPEATS}", flush=True)
    print("=" * 100, flush=True)

    model, tokenizer = load_model(cfg)

    for pos, example_index in enumerate(selected_shard, start=1):
        if example_index in done:
            print(f"[SKIP] idx={example_index} already complete", flush=True)
            continue

        print(f"[RUN] idx={example_index} local={pos}/{len(selected_shard)}", flush=True)

        try:
            result = run_one_example(
                task=task,
                ex=examples[example_index],
                example_index=example_index,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                example_dir=examples_root / f"example_{example_index:04d}",
                percent_grid=percent_grid,
            )

            append_jsonl(results_jsonl, result)

            print(
                f"[DONE] idx={example_index} "
                f"status={result['status']} "
                f"valid={result.get('metric_valid')} "
                f"norm_valid={result.get('normalization_valid')} "
                f"runtime={result.get('runtime_sec', 0):.1f}s",
                flush=True,
            )

        except Exception as e:
            err = {
                "status": "error",
                "task": cfg.TASK_NAME,
                "example_index": example_index,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_jsonl, err)
            print(f"[ERROR] idx={example_index}: {repr(e)}", flush=True)
            print(traceback.format_exc(), flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_rows = []
    if results_jsonl.exists():
        with results_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_rows.append(json.loads(line))
                except Exception:
                    pass

    summary = summarize_results(all_rows)
    summary.update(
        {
            "task": cfg.TASK_NAME,
            "shard_id": cfg.SHARD_ID,
            "num_shards": cfg.NUM_SHARDS,
            "results_jsonl": str(results_jsonl),
            "errors_jsonl": str(errors_jsonl),
        }
    )

    write_json(run_root / "summary.json", summary)

    print("[SUMMARY]", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()