#!/usr/bin/env python3
"""
04_run_duoshap_single_case.py

Run DuoShap on one long-context code-generation toy case.

Game:
  Players = sentence-level chunks of the programming problem description.

Value function:
  f(S) = pass rate of code generated from retained chunks S
         over the clean validated test suite.

DuoShap:
  For each sample:
    1. sample coalition S
    2. evaluate f(S)
    3. evaluate f(M \\ S)
    4. contribution = f(S) - f(M \\ S)
    5. add contribution to every player in S

Outputs:
  duoshap_scores_ranked.csv
  coalition_results.jsonl
  top_chunks.txt
  generation_cache/*.py
"""

import csv
import json
import os
import random
import re
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Clean configuration block
# ============================================================

CONFIG = {
    # Input
    "single_case_path": "../output/toycase_final/single_case.json",

    # Output
    "output_dir": "../output/toycase_final/duoshap_run",

    # Model
    "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "torch_dtype": "bfloat16",

    # DuoShap sampling
    "seed": 42,
    "num_samples": 100,
    "min_coalition_size": 1,
    "max_coalition_size": None,  # None means M - 1

    # Prompt/masking
    "masking_strategy": "remove",  # recommended: remove
    "include_chunk_ids_in_prompt": True,

    # Generation
    "max_new_tokens": 4096,
    "do_sample": False,
    "temperature": 0.0,

    # Execution
    "python_executable": "python",
    "timeout_seconds": 5,
    "memory_limit_mb": 512,

    # Output comparison
    "case_insensitive_output": True,

    # Caching
    "reuse_cached_generations": True,
}


# ============================================================
# IO helpers
# ============================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================
# Prompt construction
# ============================================================

def build_prompt(case: Dict[str, Any], kept_ids: List[int]) -> str:
    chunks = case["chunks"]
    kept_set = set(kept_ids)

    retained_parts = []
    for c in chunks:
        cid = c["chunk_id"]
        if cid not in kept_set:
            continue

        if CONFIG["include_chunk_ids_in_prompt"]:
            retained_parts.append(f"[Description Unit {cid}]\n{c['text']}")
        else:
            retained_parts.append(c["text"])

    description = "\n\n".join(retained_parts).strip()

    prompt = f"""
You are given a competitive programming problem. Write a correct Python 3 solution.

Requirements:
- Read all input from standard input.
- Print answers to standard output.
- Do not include explanations.
- Return only Python code.
- The solution must handle all constraints efficiently.
- Think through the algorithm internally, but output only the final Python code.

Partial problem statement:

{description}
""".strip()

    return prompt


# ============================================================
# Model generation
# ============================================================

def extract_code(text: str) -> str:
    text = text.strip()

    m = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    return text


def load_model():
    dtype = torch.bfloat16 if CONFIG["torch_dtype"] == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_code(tokenizer, model, prompt: str) -> Dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": "You are an expert competitive programmer. Output only Python code.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=CONFIG["max_new_tokens"],
        do_sample=CONFIG["do_sample"],
        temperature=CONFIG["temperature"] if CONFIG["do_sample"] else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "raw_model_output": raw_output,
        "extracted_code": extract_code(raw_output),
    }


# ============================================================
# Code execution and value function
# ============================================================

def normalize_output(text: str) -> str:
    if text is None:
        return ""

    tokens = str(text).strip().split()

    if CONFIG["case_insensitive_output"]:
        tokens = [x.lower() for x in tokens]

    return "\n".join(tokens)


def limit_resources() -> None:
    try:
        import resource

        memory_bytes = CONFIG["memory_limit_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (CONFIG["timeout_seconds"] + 1, CONFIG["timeout_seconds"] + 1),
        )
    except Exception:
        pass


def run_python_code(code: str, stdin_text: str) -> Tuple[str, str, int, bool]:
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "solution.py"
        code_path.write_text(textwrap.dedent(code).strip() + "\n", encoding="utf-8")

        try:
            proc = subprocess.run(
                [CONFIG["python_executable"], str(code_path)],
                input=stdin_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=CONFIG["timeout_seconds"],
                cwd=tmpdir,
                preexec_fn=limit_resources if os.name == "posix" else None,
            )
            return proc.stdout, proc.stderr, proc.returncode, False

        except subprocess.TimeoutExpired as e:
            return e.stdout or "", e.stderr or "", -1, True

        except Exception as e:
            return "", str(e), -1, False


def evaluate_code(code: str, tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    passed = 0
    results = []

    for test in tests:
        stdout, stderr, returncode, timed_out = run_python_code(code, test["input"])

        pred = normalize_output(stdout)
        gold = normalize_output(test["output"])

        ok = returncode == 0 and not timed_out and pred == gold

        if ok:
            passed += 1

        results.append({
            "clean_test_id": test["clean_test_id"],
            "passed": ok,
            "returncode": returncode,
            "timed_out": timed_out,
            "expected": gold,
            "predicted": pred,
            "stderr": stderr[:500],
        })

    total = len(tests)

    return {
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "test_results": results,
    }


# ============================================================
# Coalition evaluation with cache
# ============================================================

def coalition_key(kept_ids: List[int]) -> str:
    return "_".join(map(str, sorted(kept_ids)))


def evaluate_coalition(
    case: Dict[str, Any],
    kept_ids: List[int],
    tokenizer,
    model,
    generation_dir: Path,
    prompt_dir: Path,
    raw_dir: Path,
) -> Dict[str, Any]:
    key = coalition_key(kept_ids)

    code_path = generation_dir / f"coalition_{key}.py"
    prompt_path = prompt_dir / f"coalition_{key}.txt"
    raw_path = raw_dir / f"coalition_{key}.txt"

    prompt = build_prompt(case, kept_ids)
    prompt_path.write_text(prompt, encoding="utf-8")

    if CONFIG["reuse_cached_generations"] and code_path.exists():
        code = code_path.read_text(encoding="utf-8")
        raw_output = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
        generated_from_cache = True
    else:
        gen = generate_code(tokenizer, model, prompt)
        code = gen["extracted_code"]
        raw_output = gen["raw_model_output"]

        code_path.write_text(code, encoding="utf-8")
        raw_path.write_text(raw_output, encoding="utf-8")
        generated_from_cache = False

    eval_result = evaluate_code(code, case["clean_tests"])

    return {
        "kept_chunk_ids": sorted(kept_ids),
        "num_kept": len(kept_ids),
        "passed": eval_result["passed"],
        "total": eval_result["total"],
        "pass_rate": eval_result["pass_rate"],
        "generated_from_cache": generated_from_cache,
        "code_path": str(code_path),
        "prompt_path": str(prompt_path),
        "raw_output_path": str(raw_path),
        "test_results": eval_result["test_results"],
    }


# ============================================================
# DuoShap sampling
# ============================================================

def sample_coalition(all_ids: List[int]) -> List[int]:
    M = len(all_ids)
    max_k = CONFIG["max_coalition_size"]
    if max_k is None:
        max_k = M - 1

    k = random.randint(CONFIG["min_coalition_size"], max_k)
    return sorted(random.sample(all_ids, k))


def write_scores_csv(path: Path, case: Dict[str, Any], sums: List[float], counts: List[int]) -> List[Dict[str, Any]]:
    rows = []

    for c in case["chunks"]:
        cid = c["chunk_id"]
        score = sums[cid] / counts[cid] if counts[cid] > 0 else 0.0

        rows.append({
            "rank": None,
            "chunk_id": cid,
            "title": c.get("title", f"Chunk {cid}"),
            "num_words": c.get("num_words", ""),
            "duoshap_score": score,
            "score_sum": sums[cid],
            "count": counts[cid],
            "text": c["text"],
        })

    rows.sort(key=lambda r: r["duoshap_score"], reverse=True)

    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "chunk_id",
                "title",
                "num_words",
                "duoshap_score",
                "score_sum",
                "count",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def write_top_chunks(path: Path, rows: List[Dict[str, Any]], top_k: int = 15) -> None:
    lines = []
    lines.append("Top DuoShap chunks\n")
    lines.append("=" * 80 + "\n")

    for r in rows[:top_k]:
        lines.append(f"Rank {r['rank']} | chunk_id={r['chunk_id']} | score={r['duoshap_score']:.6f} | count={r['count']}\n")
        lines.append(r["text"] + "\n")
        lines.append("-" * 80 + "\n")

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    random.seed(CONFIG["seed"])

    output_dir = Path(CONFIG["output_dir"])
    generation_dir = output_dir / "generation_cache"
    prompt_dir = output_dir / "prompt_cache"
    raw_dir = output_dir / "raw_output_cache"
    results_dir = output_dir / "results"

    for d in [output_dir, generation_dir, prompt_dir, raw_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    coalition_results_path = results_dir / "coalition_results.jsonl"
    scores_path = results_dir / "duoshap_scores_ranked.csv"
    top_chunks_path = results_dir / "top_chunks.txt"
    config_path = results_dir / "run_config.json"

    if coalition_results_path.exists():
        coalition_results_path.unlink()

    write_json(config_path, CONFIG)

    print("=" * 80)
    print("Run DuoShap on Single Code-Generation Toy Case")
    print("=" * 80)

    case = load_json(CONFIG["single_case_path"])
    M = case["num_chunks"]
    all_ids = list(range(M))

    print(f"case_id: {case['case_id']}")
    print(f"players: {M}")
    print(f"clean tests: {case['num_clean_tests']}")
    print(f"num_samples: {CONFIG['num_samples']}")
    print(f"expected LLM generations: {2 * CONFIG['num_samples']}")
    print("=" * 80)

    print("[INFO] Loading model...")
    tokenizer, model = load_model()

    sums = [0.0 for _ in range(M)]
    counts = [0 for _ in range(M)]

    start_time = time.time()

    for sample_id in range(CONFIG["num_samples"]):
        S = sample_coalition(all_ids)
        S_set = set(S)
        comp = sorted([i for i in all_ids if i not in S_set])

        print("\n" + "-" * 80)
        print(f"[Sample {sample_id + 1}/{CONFIG['num_samples']}] |S|={len(S)} |comp|={len(comp)}")

        result_S = evaluate_coalition(
            case=case,
            kept_ids=S,
            tokenizer=tokenizer,
            model=model,
            generation_dir=generation_dir,
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
        )

        result_comp = evaluate_coalition(
            case=case,
            kept_ids=comp,
            tokenizer=tokenizer,
            model=model,
            generation_dir=generation_dir,
            prompt_dir=prompt_dir,
            raw_dir=raw_dir,
        )

        contribution = result_S["pass_rate"] - result_comp["pass_rate"]

        for i in S:
            sums[i] += contribution
            counts[i] += 1

        sample_record = {
            "sample_id": sample_id,
            "S": {
                "kept_chunk_ids": S,
                "num_kept": len(S),
                "passed": result_S["passed"],
                "total": result_S["total"],
                "pass_rate": result_S["pass_rate"],
                "code_path": result_S["code_path"],
                "prompt_path": result_S["prompt_path"],
                "raw_output_path": result_S["raw_output_path"],
                "generated_from_cache": result_S["generated_from_cache"],
            },
            "complement": {
                "kept_chunk_ids": comp,
                "num_kept": len(comp),
                "passed": result_comp["passed"],
                "total": result_comp["total"],
                "pass_rate": result_comp["pass_rate"],
                "code_path": result_comp["code_path"],
                "prompt_path": result_comp["prompt_path"],
                "raw_output_path": result_comp["raw_output_path"],
                "generated_from_cache": result_comp["generated_from_cache"],
            },
            "duoshap_contribution": contribution,
        }

        append_jsonl(coalition_results_path, sample_record)

        print(f"f(S)={result_S['pass_rate']:.4f} ({result_S['passed']}/{result_S['total']})")
        print(f"f(comp)={result_comp['pass_rate']:.4f} ({result_comp['passed']}/{result_comp['total']})")
        print(f"contribution={contribution:.4f}")

        # Periodic score checkpoint
        if (sample_id + 1) % 10 == 0 or (sample_id + 1) == CONFIG["num_samples"]:
            rows = write_scores_csv(scores_path, case, sums, counts)
            write_top_chunks(top_chunks_path, rows, top_k=15)
            print(f"[CHECKPOINT] saved ranked scores after {sample_id + 1} samples")

    elapsed = time.time() - start_time

    rows = write_scores_csv(scores_path, case, sums, counts)
    write_top_chunks(top_chunks_path, rows, top_k=20)

    final_summary = {
        "case_id": case["case_id"],
        "num_players": M,
        "num_clean_tests": case["num_clean_tests"],
        "num_samples": CONFIG["num_samples"],
        "elapsed_seconds": elapsed,
        "scores_csv": str(scores_path),
        "top_chunks_txt": str(top_chunks_path),
        "coalition_results_jsonl": str(coalition_results_path),
        "top_10_chunks": rows[:10],
    }

    write_json(results_dir / "final_summary.json", final_summary)

    print("\n" + "=" * 80)
    print("[DONE] DuoShap finished")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Scores: {scores_path}")
    print(f"Top chunks: {top_chunks_path}")
    print(f"Coalitions: {coalition_results_path}")
    print("=" * 80)

    print("\nTop 10 chunks:")
    for r in rows[:10]:
        print(f"Rank {r['rank']:>2} | chunk {r['chunk_id']:>3} | score={r['duoshap_score']:.6f} | count={r['count']}")
        print(f"  {r['text'][:180]}")


if __name__ == "__main__":
    main()