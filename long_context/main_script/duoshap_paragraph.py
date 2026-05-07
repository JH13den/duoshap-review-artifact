#!/usr/bin/env python3
"""
DuoShap LongBench paragraph/passage-level runner.

Supported tasks:
  - passage_retrieval_en
  - hotpotqa
  - 2wikimqa

Purpose:
  - Run DuoShap over many examples.
  - Support Slurm array sharding.
  - Save per-example Shapley values and summary results.
  - Keep MC baseline in a separate script later.

Important:
  - No mini-batch.
  - No flash attention.
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
# DEFAULT CONFIG
# You can override most of these from command line / Slurm.
# ============================================================

@dataclass
class Config:
    HF_HOME: Path = Path("xxxxxxxxxxxx")
    LONGBENCH_DATA_DIR: Path = Path(
        "xxxx"
    )
    OUT_DIR: Path = Path(
        "xxxxxxxx"
    )

    RUN_NAME: str = "duoshap_qwen25_7b_paragraph_level"

    MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"
    DTYPE: str = "fp16"
    USE_CHAT_TEMPLATE: bool = True
    USE_MODEL_CACHE: bool = False

    TASK_NAME: str = "passage_retrieval_en"

    EXAMPLE_START: int = 0
    EXAMPLE_END: Optional[int] = None
    MAX_EXAMPLES: Optional[int] = 1

    NUM_SHARDS: int = 1
    SHARD_ID: int = 0

    SEED: int = 42

    OMIT_TOKEN: str = "[OMITTED]"
    ENABLE_VALUE_CACHE: bool = True
    SAVE_VALUE_CACHE: bool = True
    MAX_PROMPT_TOKENS_WARN: int = 40000

    # Main DuoShap hyperparameter.
    NUM_GROUPS_PER_PLAYER: int = 3
    USE_ANTITHETIC: bool = True
    RENORMALIZE_SUM: bool = True

    TOPK_TO_SAVE: int = 5

    # QA answer-containment metric settings.
    QA_MIN_ANSWER_CHARS_FOR_SUPPORT: int = 3
    QA_EXCLUDE_YES_NO: bool = True

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

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default=cfg.TASK_NAME,
                        choices=["passage_retrieval_en", "hotpotqa", "2wikimqa"])
    parser.add_argument("--run-name", type=str, default=cfg.RUN_NAME)

    parser.add_argument("--hf-home", type=Path, default=cfg.HF_HOME)
    parser.add_argument("--data-dir", type=Path, default=cfg.LONGBENCH_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=cfg.OUT_DIR)

    parser.add_argument("--model-id", type=str, default=cfg.MODEL_ID)
    parser.add_argument("--dtype", type=str, default=cfg.DTYPE, choices=["fp16", "fp32"])

    parser.add_argument("--example-start", type=int, default=cfg.EXAMPLE_START)
    parser.add_argument("--example-end", type=parse_optional_int, default=cfg.EXAMPLE_END)
    parser.add_argument("--max-examples", type=parse_optional_int, default=cfg.MAX_EXAMPLES)

    parser.add_argument("--num-shards", type=int, default=cfg.NUM_SHARDS)
    parser.add_argument("--shard-id", type=int, default=cfg.SHARD_ID)

    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--num-groups-per-player", type=int, default=cfg.NUM_GROUPS_PER_PLAYER)

    parser.add_argument("--no-antithetic", action="store_true")
    parser.add_argument("--no-renormalize", action="store_true")
    parser.add_argument("--no-value-cache", action="store_true")
    parser.add_argument("--no-save-value-cache", action="store_true")
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    cfg.TASK_NAME = args.task
    cfg.RUN_NAME = args.run_name

    cfg.HF_HOME = args.hf_home
    cfg.LONGBENCH_DATA_DIR = args.data_dir
    cfg.OUT_DIR = args.out_dir

    cfg.MODEL_ID = args.model_id
    cfg.DTYPE = args.dtype

    cfg.EXAMPLE_START = args.example_start
    cfg.EXAMPLE_END = args.example_end
    cfg.MAX_EXAMPLES = args.max_examples

    cfg.NUM_SHARDS = args.num_shards
    cfg.SHARD_ID = args.shard_id

    cfg.SEED = args.seed
    cfg.NUM_GROUPS_PER_PLAYER = args.num_groups_per_player

    cfg.USE_ANTITHETIC = not args.no_antithetic
    cfg.RENORMALIZE_SUM = not args.no_renormalize
    cfg.ENABLE_VALUE_CACHE = not args.no_value_cache
    cfg.SAVE_VALUE_CACHE = not args.no_save_value_cache
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
        indices = indices[:cfg.MAX_EXAMPLES]

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


def stable_seed(base_seed: int, task_name: str, example_index: int) -> int:
    h = sum((i + 1) * ord(c) for i, c in enumerate(task_name))
    return int((base_seed + 1009 * example_index + 9173 * h) % (2**32 - 1))


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
    """
    Returns:
      support_ids: 0-based support indices
      metric_valid: whether this example counts in hit-rate denominator
      reason
    """
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
    def avg_logprob_gold(self, plain_prompt: str) -> float:
        prompt = maybe_chat(self.tokenizer, plain_prompt, self.cfg.USE_CHAT_TEMPLATE)
        dev = model_device(self.model)

        prompt_ids = self.tokenizer(
            prompt,
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
        prompt = self.task.build_prompt(context, self.query)

        wrapped = maybe_chat(self.tokenizer, prompt, self.cfg.USE_CHAT_TEMPLATE)
        token_len = self.tokenizer(
            wrapped,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.shape[1]

        if token_len > self.cfg.MAX_PROMPT_TOKENS_WARN:
            print(
                f"[WARN] prompt token length={token_len} exceeds {self.cfg.MAX_PROMPT_TOKENS_WARN}",
                flush=True,
            )

        value = self.avg_logprob_gold(prompt)

        if self.cfg.ENABLE_VALUE_CACHE:
            self.cache[mask] = value

        return value


# ============================================================
# DUOSHAP
# ============================================================

def group_shapley_value(value_fn: LongContextValueFn, mask: int, f_empty: float, f_full: float) -> float:
    comp = value_fn.full_mask ^ mask
    return 0.5 * ((f_full - value_fn.f(comp)) + (value_fn.f(mask) - f_empty))


def run_duoshap_conditioned(
    value_fn: LongContextValueFn,
    cfg: Config,
    rng: np.random.Generator,
    f_empty: float,
    f_full: float,
) -> Tuple[np.ndarray, np.ndarray]:
    M = value_fn.M
    full = value_fn.full_mask

    acc = np.zeros(M, dtype=np.float64)
    cnt = np.zeros(M, dtype=np.int64)

    for i in range(M):
        others = np.array([j for j in range(M) if j != i], dtype=np.int64)

        for _ in range(cfg.NUM_GROUPS_PER_PLAYER):
            K = int(rng.integers(1, M + 1))

            if K == 1:
                chosen = np.array([i], dtype=np.int64)
            else:
                sampled = rng.choice(others, size=K - 1, replace=False)
                chosen = np.concatenate([np.array([i], dtype=np.int64), sampled])

            mask = 0
            for j in chosen.tolist():
                mask |= (1 << int(j))

            g = group_shapley_value(value_fn, mask, f_empty, f_full)
            acc[i] += g
            cnt[i] += 1

            if cfg.USE_ANTITHETIC:
                anti = (full ^ mask) | (1 << i)
                g2 = group_shapley_value(value_fn, anti, f_empty, f_full)
                acc[i] += g2
                cnt[i] += 1

    phi = acc / np.maximum(cnt, 1)

    if cfg.RENORMALIZE_SUM:
        target = f_full - f_empty
        phi = phi + (target - float(phi.sum())) / M

    return phi, cnt


# ============================================================
# OUTPUT HELPERS
# ============================================================

def write_shapley_csv(
    path: Path,
    phi: np.ndarray,
    counts: np.ndarray,
    players: List[str],
    support_ids: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort(-phi)
    support = set(support_ids)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "player_index_1based",
                "phi",
                "count",
                "is_support",
                "text_preview",
            ],
        )
        writer.writeheader()

        for rank, idx in enumerate(order.tolist(), start=1):
            text = players[idx].replace("\n", " ")
            if len(text) > 500:
                text = text[:500] + "..."

            writer.writerow(
                {
                    "rank": rank,
                    "player_index_1based": idx + 1,
                    "phi": float(phi[idx]),
                    "count": int(counts[idx]),
                    "is_support": bool(idx in support),
                    "text_preview": text,
                }
            )


def topk_rows(phi: np.ndarray, support_ids: Sequence[int], k: int) -> List[dict]:
    support = set(support_ids)
    order = np.argsort(-phi)[:k]

    rows = []
    for rank, idx in enumerate(order.tolist(), start=1):
        rows.append(
            {
                "rank": rank,
                "player_index_1based": idx + 1,
                "phi": float(phi[idx]),
                "is_support": bool(idx in support),
            }
        )

    return rows


def summarize(rows: List[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    valid = [r for r in ok if r.get("metric_valid") is True]
    hits = [r for r in valid if r.get("top1_hit") is True]

    runtimes = [r["runtime_sec"] for r in ok]

    return {
        "num_ok": len(ok),
        "num_metric_valid": len(valid),
        "num_hits": len(hits),
        "hit_rate_valid_only": len(hits) / len(valid) if valid else None,
        "metric_coverage": len(valid) / len(ok) if ok else None,
        "avg_runtime_sec": float(np.mean(runtimes)) if runtimes else None,
        "std_runtime_sec": float(np.std(runtimes)) if runtimes else None,
    }


# ============================================================
# ONE EXAMPLE
# ============================================================

def run_one_example(
    task: TaskSpec,
    ex: dict,
    example_index: int,
    model,
    tokenizer,
    cfg: Config,
    example_dir: Path,
) -> dict:
    start_time = time.time()

    query = ex["input"]
    context = ex["context"]
    answers = ex["answers"]

    gold_answer = answers[0]

    players = task.split_players(context)

    support_ids, metric_valid, metric_reason = get_support_ids(
        task=task,
        players=players,
        answers=answers,
        cfg=cfg,
    )

    value_fn = LongContextValueFn(
        model=model,
        tokenizer=tokenizer,
        task=task,
        players=players,
        query=query,
        gold_answer=gold_answer,
        cfg=cfg,
    )

    f_empty = value_fn.f(0)
    f_full = value_fn.f(value_fn.full_mask)

    rng = np.random.default_rng(stable_seed(cfg.SEED, task.name, example_index))

    phi, counts = run_duoshap_conditioned(
        value_fn=value_fn,
        cfg=cfg,
        rng=rng,
        f_empty=f_empty,
        f_full=f_full,
    )

    top_idx = int(np.argmax(phi))
    top1_hit = bool(metric_valid and top_idx in set(support_ids))

    runtime = time.time() - start_time

    example_dir.mkdir(parents=True, exist_ok=True)

    write_shapley_csv(
        example_dir / "shapley.csv",
        phi=phi,
        counts=counts,
        players=players,
        support_ids=support_ids,
    )

    with (example_dir / "top_player_text.txt").open("w", encoding="utf-8") as f:
        f.write(players[top_idx])

    if cfg.SAVE_VALUE_CACHE:
        write_json(
            example_dir / "value_cache.json",
            {
                "subset_mask_to_value": {str(k): float(v) for k, v in value_fn.cache.items()},
            },
        )

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
        "num_players": len(players),
        "support_ids_1based": [i + 1 for i in support_ids],
        "top_player_1based": top_idx + 1,
        "top_score": float(phi[top_idx]),
        "top1_hit": top1_hit,
        "topk": topk_rows(phi, support_ids, cfg.TOPK_TO_SAVE),
        "f_empty": float(f_empty),
        "f_full": float(f_full),
        "f_full_minus_empty": float(f_full - f_empty),
        "phi_sum": float(phi.sum()),
        "runtime_sec": float(runtime),
        "value_cache_size": len(value_fn.cache),
        "num_groups_per_player": cfg.NUM_GROUPS_PER_PLAYER,
        "use_antithetic": cfg.USE_ANTITHETIC,
        "renormalize_sum": cfg.RENORMALIZE_SUM,
    }

    write_json(example_dir / "result.json", result)

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

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    data_path = cfg.LONGBENCH_DATA_DIR / f"{cfg.TASK_NAME}.jsonl"
    examples = load_jsonl(data_path)

    selected = select_indices(len(examples), cfg)
    selected_shard = shard_indices(selected, cfg.SHARD_ID, cfg.NUM_SHARDS)

    run_root = cfg.OUT_DIR / cfg.RUN_NAME / cfg.TASK_NAME / f"shard_{cfg.SHARD_ID:03d}_of_{cfg.NUM_SHARDS:03d}"
    examples_root = run_root / "examples"

    run_root.mkdir(parents=True, exist_ok=True)

    write_json(
        run_root / "config.json",
        {
            "config": asdict(cfg),
            "num_examples_in_file": len(examples),
            "selected_indices_before_shard": selected,
            "selected_indices_this_shard": selected_shard,
        },
    )

    results_jsonl = run_root / "results.jsonl"
    errors_jsonl = run_root / "errors.jsonl"

    done = completed_indices(results_jsonl) if cfg.RESUME_SKIP_COMPLETED else set()

    print("=" * 80, flush=True)
    print(f"[TASK] {cfg.TASK_NAME}", flush=True)
    print(f"[DATA] {data_path}", flush=True)
    print(f"[OUT]  {run_root}", flush=True)
    print(f"[INFO] examples in file: {len(examples)}", flush=True)
    print(f"[INFO] selected before shard: {len(selected)}", flush=True)
    print(f"[INFO] shard {cfg.SHARD_ID}/{cfg.NUM_SHARDS}: {len(selected_shard)} examples", flush=True)
    print(f"[INFO] num_groups_per_player={cfg.NUM_GROUPS_PER_PLAYER}", flush=True)
    print("=" * 80, flush=True)

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
            )

            append_jsonl(results_jsonl, result)

            print(
                f"[DONE] idx={example_index} "
                f"top={result['top_player_1based']} "
                f"hit={result['top1_hit']} "
                f"valid={result['metric_valid']} "
                f"runtime={result['runtime_sec']:.1f}s",
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

    summary = summarize(all_rows)
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