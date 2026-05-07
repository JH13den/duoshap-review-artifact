#!/usr/bin/env python3
"""
Unified Global SODA (Eq. 3) runner (GT/IOI semantics) for the unified cache layout:

  ../cache_output/{TASK_KEY}/{MODEL_KEY}/nodes/{split}/batch_{b}/
    head_L{L}_H{h}.pt   [B, T, d_model]  (post-W_O head writes; NO b_O)
    mlp_L{L}.pt         [B, T, d_model]
    batch_index.json
  ../cache_output/{TASK_KEY}/{MODEL_KEY}/nodes/meta.json   (written by your cache script)

Guarantees:
- Patch ONLY last position (last non-pad token)
- Patch points:
    blocks.{L}.attn.hook_result   [B,T,d_model]
    blocks.{L}.hook_mlp_out       [B,T,d_model]
- f(S) = -E_x[ KL( p_clean || p_patched ) ] at last token
- Eq.3 group Shapley with antithetic complements (S and M\\S)
- Selection = Top-N per layer OR global percentile
- Outputs:
    OUT_ROOT / OUT_GROUP / out_tag / ...

CRITICAL FIXES INCLUDED:
1) LEFT padding correctness:
   - last_positions() works for left/right padding (uses mask*pos max)
2) Cache/eval alignment:
   - tokenization pads to the cached batch T (so cached activations indices match hook activations)
3) TLens compat:
   - tl_call / tl_call_with_hooks tries attention_mask, then padding_mask, then no mask
4) attn.hook_result exactness:
   - cached head writes are z @ W_O per head (no b_O)
   - hook_result includes + b_O, so we add b_O back when constructing attn_mix

IOI FIX (THIS VERSION):
- Pythia tokenizer may split names like " Zoe" into multiple tokens.
- IOI evaluation now supports multi-token target/distractor by using
  mean logprob over the token sequence at the last positions.
- Keeps output keys identical:
    mean_logprob_target, cross_entropy, mean_logit_diff
  (mean_logit_diff = mean_logprob(target_seq) - mean_logprob(distractor_seq))

Usage:
  python duoshap.py --task gt
  python duoshap.py --task ioi
  python duoshap.py --task docstring
  python duoshap.py --task all
  python duoshap.py --task gt ioi
"""

import os, json, csv, random, time, re, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer


# ============================ OUTPUT LAYOUT ===================================
OUT_ROOT: str = "../eval_output"
OUT_GROUP: str = "Duoshap_pythia"
# ==============================================================================


# ================================ CONFIG ======================================
CONFIG = {
    "MODEL_KEY": "pythia-350m",
    "TLENS_MODEL_NAME": "EleutherAI/pythia-350m",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "float32",
    "CACHE_OUTPUT_ROOT": "../cache_output",

    "TASKS": {
        "gt": {
            "out_tag": "gt",
            "clean_jsonl": "../data/gt/gt_clean.jsonl",

            "num_layers": 24,
            "num_heads": 16,
            "include_mlp": True,
            "nodes_per_layer": 17,  # 16 heads + 1 mlp
            "batch_size": 32,

            "repeats": 400,
            "k_min": 10,
            "k_max": 100,
            "top_n_per_layer": 12,
            "global_percentile": 0.90,

            "task_type": "gt",
            "seed": 123,
        },
        "ioi": {
            "out_tag": "ioi",
            "clean_jsonl": "../data/ioi/ioi_clean.jsonl",

            "num_layers": 24,
            "num_heads": 16,
            "include_mlp": True,
            "nodes_per_layer": 17,
            "batch_size": 32,

            "repeats": 400,
            "k_min": 10,
            "k_max": 100,
            "top_n_per_layer": 12,
            "global_percentile": 0.90,

            "task_type": "ioi",
            "seed": 123,
            "debug_sanity_all_clean": True,
        },
        "docstring": {
            "out_tag": "docstring",
            "clean_jsonl": "../data/docstring/doc_clean.jsonl",

            "num_layers": 24,
            "num_heads": 16,
            "include_mlp": True,
            "nodes_per_layer": 17,
            "batch_size": 32,

            "repeats": 400,
            "k_min": 10,
            "k_max": 100,
            "top_n_per_layer": 12,
            "global_percentile": 0.90,

            "task_type": "docstring",
            "seed": 123,
        },
    },
}
# ==============================================================================


# ============================== HELPERS =======================================
DIGIT_RE = re.compile(r"^\d{2}$")

def set_seed(seed: int) -> None:
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]

def batches(n: int, bs: int):
    nb = (n + bs - 1) // bs
    for b in range(nb):
        s = b * bs
        e = min(n, s + bs)
        yield b, s, e

def last_positions(attn_mask: torch.Tensor) -> List[int]:
    """
    Correct for BOTH left- and right-padding.
    Returns last index where attention_mask == 1.
    """
    T = attn_mask.size(1)
    pos = torch.arange(T, device=attn_mask.device).unsqueeze(0)  # [1,T]
    last = (attn_mask * pos).max(dim=1).values.long()
    return last.tolist()

def describe_node(i: int, num_heads: int) -> str:
    return f"head_{i}" if i < num_heads else "mlp"

def token_id_single(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"String '{s}' is not a single token (ids={ids}).")
    return ids[0]

def ensure_out_dir(out_tag: str) -> Path:
    out_dir = Path(OUT_ROOT) / OUT_GROUP / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def resolve_cache_nodes_root(task_key: str, model_key: str) -> Path:
    return Path(CONFIG["CACHE_OUTPUT_ROOT"]) / task_key / model_key / "nodes"

def torch_dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float32

def read_cache_meta(nodes_root: Path) -> Optional[Dict[str, Any]]:
    p = nodes_root / "meta.json"
    if not p.exists():
        return None
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def cache_batch_shape(
    nodes_root: Path,
    split: str,
    batch_idx: int,
    include_mlp: bool,
) -> Tuple[int, int]:
    """
    Returns (B_cache, T_cache) for this cached batch without scanning all layers.
    """
    bdir = nodes_root / split / f"batch_{batch_idx}"
    probe = None
    if include_mlp and (bdir / "mlp_L0.pt").exists():
        probe = bdir / "mlp_L0.pt"
    else:
        probe = bdir / "head_L0_H0.pt"
    x = torch.load(probe, map_location="cpu")
    Bc, Tc = int(x.shape[0]), int(x.shape[1])
    del x
    return Bc, Tc

def tokenize_to_cache_len(
    tok,
    prompts: List[str],
    T_cache: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize prompts padded to EXACTLY max_length=T_cache.
    This is REQUIRED so that "last position index" matches cached activations indices.
    """
    raw = tok(prompts, padding=False, truncation=False)
    max_len = max(len(ids) for ids in raw["input_ids"]) if len(raw["input_ids"]) else 0
    if max_len > T_cache:
        raise RuntimeError(f"[tokenize_to_cache_len] max_len={max_len} > T_cache={T_cache}. Cache/eval misaligned.")
    enc = tok(prompts, return_tensors="pt", padding="max_length", max_length=T_cache, truncation=False)
    return enc


# ======================== TLENS COMPAT HELPERS ================================
def tl_call(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
):
    """
    Compat: try attention_mask, then padding_mask, then no mask.
    """
    try:
        return model(input_ids, attention_mask=attn_mask)
    except TypeError:
        pass
    if attn_mask is not None:
        try:
            return model(input_ids, padding_mask=attn_mask.to(dtype=torch.bool, device=input_ids.device))
        except TypeError:
            pass
    return model(input_ids)

def tl_call_with_hooks(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    fwd_hooks,
    attn_mask: Optional[torch.Tensor] = None,
):
    """
    Compat: try attention_mask, then padding_mask, then no mask.
    """
    try:
        return model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks, attention_mask=attn_mask)
    except TypeError:
        pass
    if attn_mask is not None:
        try:
            return model.run_with_hooks(
                input_ids,
                fwd_hooks=fwd_hooks,
                padding_mask=attn_mask.to(dtype=torch.bool, device=input_ids.device),
            )
        except TypeError:
            pass
    return model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)


# ============================ TASK METRICS ====================================
def get_batch_target_ids_gt(records: List[Dict[str, Any]], tok) -> torch.Tensor:
    ids = [token_id_single(tok, r["target_str"]) for r in records]
    return torch.tensor(ids, dtype=torch.long)

def get_batch_distractor_ids_gt(records: List[Dict[str, Any]], tok) -> torch.Tensor:
    ds = []
    for r in records:
        s = r.get("distractor_str", f"{int(r['yy_start']):02d}")
        ds.append(token_id_single(tok, s))
    return torch.tensor(ds, dtype=torch.long)

def build_two_digit_vocab(tok):
    yy_to_id = {}
    for yy in range(100):
        s = f"{yy:02d}"
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            yy_to_id[yy] = enc[0]
    return yy_to_id

def get_batch_ids_ioi(records: List[Dict[str, Any]], tok):
    # NOTE: kept for compatibility, but IOI eval no longer requires single-token strings.
    tgt, dis = [], []
    for r in records:
        tgt.append(token_id_single(tok, r["target_str"]))
        dis.append(token_id_single(tok, r["distractor_str"]))
    return torch.tensor(tgt, dtype=torch.long), torch.tensor(dis, dtype=torch.long)


# ============================ CACHE IO ========================================
def load_batch_cache_for_layer(
    nodes_root: Path,
    split: str,
    batch_idx: int,
    L: int,
    H: int,
    include_mlp: bool,
    device: torch.device,
    dtype: torch.dtype,
):
    bdir = nodes_root / split / f"batch_{batch_idx}"
    head_writes: List[torch.Tensor] = []
    for h in range(H):
        x = torch.load(bdir / f"head_L{L}_H{h}.pt", map_location="cpu")
        head_writes.append(x.to(device=device, dtype=dtype, non_blocking=True))
    mlp_out = None
    if include_mlp:
        mlp_out = torch.load(bdir / f"mlp_L{L}.pt", map_location="cpu").to(
            device=device, dtype=dtype, non_blocking=True
        )
    return head_writes, mlp_out


# ============================ PATCHING (ALIGNED) ==============================
def mix_lastpos_heads(last_pos, keep_clean_nodes, heads_clean, heads_corr):
    device = heads_clean[0].device
    B = heads_clean[0].shape[0]
    d_model = heads_clean[0].shape[-1]
    out = torch.zeros(B, d_model, device=device, dtype=heads_clean[0].dtype)
    lp_idx = torch.as_tensor(last_pos, device=device, dtype=torch.long)
    ar = torch.arange(B, device=device)
    for h in range(len(heads_clean)):
        src = heads_clean[h] if (h in keep_clean_nodes) else heads_corr[h]
        out += src[ar, lp_idx, :]
    return out

def mix_lastpos_mlp(last_pos, keep_clean_nodes, mlp_clean, mlp_corr, mlp_idx):
    if mlp_clean is None or mlp_corr is None:
        ref = mlp_clean if mlp_clean is not None else mlp_corr
        return torch.zeros_like(ref[:, 0, :])
    device = mlp_clean.device
    B = mlp_clean.shape[0]
    lp_idx = torch.as_tensor(last_pos, device=device, dtype=torch.long)
    ar = torch.arange(B, device=device)
    use_clean = (mlp_idx in keep_clean_nodes)
    src = mlp_clean if use_clean else mlp_corr
    return src[ar, lp_idx, :]

def make_overwrite_lastpos_hook(last_pos, repl_rows_Bd):
    B = repl_rows_Bd.shape[0]
    idxs = None
    def _hook(act, hook):
        nonlocal idxs
        if idxs is None or idxs.device != act.device:
            idxs = torch.as_tensor(last_pos, device=act.device, dtype=torch.long)
        ar = torch.arange(B, device=act.device)
        act = act.clone()
        act[ar, idxs, :] = repl_rows_Bd.to(act.device, dtype=act.dtype)
        return act
    return _hook

@torch.no_grad()
def build_hooks_for_layers(
    model: HookedTransformer,
    tok_batch: Dict[str, torch.Tensor],
    layers_to_patch: List[int],
    per_layer_keep_sets: Dict[int, Set[int]],
    nodes_root: Path,
    batch_idx: int,
    H: int,
    include_mlp: bool,
    device: torch.device,
    dtype: torch.dtype,
):
    attn_mask = tok_batch["attention_mask"].to(device)
    last_pos = last_positions(attn_mask)  # correct for left padding
    hooks: List[Tuple[str, Any]] = []

    for L in layers_to_patch:
        heads_clean, mlp_clean = load_batch_cache_for_layer(
            nodes_root, "clean", batch_idx, L, H, include_mlp, device, dtype
        )
        heads_corr, mlp_corr = load_batch_cache_for_layer(
            nodes_root, "corrupted", batch_idx, L, H, include_mlp, device, dtype
        )
        keep = per_layer_keep_sets.get(L, set())

        attn_mix = mix_lastpos_heads(last_pos, keep, heads_clean, heads_corr)

        # IMPORTANT: hook_result includes b_O; cached head writes do NOT. Add it back.
        attn_mod = model.blocks[L].attn
        b_O = getattr(attn_mod, "b_O", None)
        if b_O is not None:
            attn_mix = attn_mix + b_O.to(device=device, dtype=attn_mix.dtype)[None, :]

        mlp_mix  = mix_lastpos_mlp(last_pos, keep, mlp_clean, mlp_corr, mlp_idx=H)

        hooks.append((f"blocks.{L}.attn.hook_result", make_overwrite_lastpos_hook(last_pos, attn_mix)))
        hooks.append((f"blocks.{L}.hook_mlp_out",     make_overwrite_lastpos_hook(last_pos, mlp_mix)))

        del heads_clean, heads_corr, mlp_clean, mlp_corr, attn_mix, mlp_mix

    return hooks


# ============================ VALUE FUNCTION ==================================
@torch.no_grad()
def dataset_value_global(
    model: HookedTransformer,
    tok,
    clean_recs: List[Dict[str, Any]],
    nodes_root: Path,
    S_global: Set[int],
    gid_to_Li: List[Tuple[int, int]],
    H: int,
    include_mlp: bool,
    bs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    n_layers = model.cfg.n_layers
    per_layer_keep: Dict[int, Set[int]] = {L: set() for L in range(n_layers)}
    for gid in S_global:
        L, i = gid_to_Li[gid]
        per_layer_keep[L].add(i)

    total_kl, total_B = 0.0, 0
    all_layers = list(range(n_layers))
    N = len(clean_recs)

    for bidx, s, e in batches(N, bs):
        recs_batch = clean_recs[s:e]
        prompts = [r["prompt"] for r in recs_batch]

        # ✅ Align eval tokenization length to cached batch T
        Bc, Tc = cache_batch_shape(nodes_root, "clean", bidx, include_mlp=include_mlp)
        if Bc != len(prompts):
            raise RuntimeError(f"[cache/eval mismatch] batch_{bidx}: cache B={Bc} vs eval B={len(prompts)}. Check batch_size + dataset order.")
        tok_batch = tokenize_to_cache_len(tok, prompts, Tc)

        input_ids = tok_batch["input_ids"].to(device)
        attn_mask = tok_batch["attention_mask"].to(device)
        last_pos = last_positions(attn_mask)
        B = input_ids.size(0)

        logits_clean = tl_call(model, input_ids, attn_mask)

        hooks = build_hooks_for_layers(
            model, tok_batch, all_layers, per_layer_keep,
            nodes_root, bidx, H, include_mlp, device, dtype
        )
        logits_patched = tl_call_with_hooks(model, input_ids, hooks, attn_mask)

        idx = torch.as_tensor(last_pos, device=device, dtype=torch.long)
        lp = logits_clean[torch.arange(B, device=device), idx, :]
        lq = logits_patched[torch.arange(B, device=device), idx, :]

        logp = F.log_softmax(lp, dim=-1)
        logq = F.log_softmax(lq, dim=-1)
        p = logp.exp()
        kl = (p * (logp - logq)).sum(dim=-1).mean().item()

        total_kl += kl * B
        total_B  += B

        del logits_clean, logits_patched, lp, lq, logp, logq, p
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return - (total_kl / total_B)


# ============================ EVALUATION (GT) =================================
@torch.no_grad()
def evaluate_gt(
    model: HookedTransformer,
    tok,
    clean_recs: List[Dict[str, Any]],
    nodes_root: Path,
    keep_sets: Dict[int, Set[int]],
    H: int,
    include_mlp: bool,
    bs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    yy_to_id = build_two_digit_vocab(tok)
    yy_ids = torch.tensor(list(yy_to_id.values()), device=device) if yy_to_id else torch.tensor([], device=device)
    yy_vals = torch.tensor(list(yy_to_id.keys()), device=device, dtype=torch.long) if yy_to_id else torch.tensor([], device=device)

    def _core(per_layer_keep: Dict[int, Set[int]]) -> Dict[str, Any]:
        total_kl, total_B = 0.0, 0
        sum_logprob, sum_logitdiff, sum_pd, count_pd = 0.0, 0.0, 0.0, 0

        n_layers = model.cfg.n_layers
        all_layers = list(range(n_layers))
        N = len(clean_recs)

        for bidx, s, e in batches(N, bs):
            recs_batch = clean_recs[s:e]
            prompts = [r["prompt"] for r in recs_batch]

            Bc, Tc = cache_batch_shape(nodes_root, "clean", bidx, include_mlp=include_mlp)
            if Bc != len(prompts):
                raise RuntimeError(f"[cache/eval mismatch] batch_{bidx}: cache B={Bc} vs eval B={len(prompts)}")
            tok_batch = tokenize_to_cache_len(tok, prompts, Tc)

            input_ids = tok_batch["input_ids"].to(device)
            attn_mask = tok_batch["attention_mask"].to(device)
            last_pos = last_positions(attn_mask)
            B = input_ids.size(0)

            target_ids     = get_batch_target_ids_gt(recs_batch, tok).to(device)
            distractor_ids = get_batch_distractor_ids_gt(recs_batch, tok).to(device)

            logits_clean = tl_call(model, input_ids, attn_mask)
            hooks = build_hooks_for_layers(
                model, tok_batch, all_layers, per_layer_keep,
                nodes_root, bidx, H, include_mlp, device, dtype
            )
            logits_patched = tl_call_with_hooks(model, input_ids, hooks, attn_mask)

            idx = torch.as_tensor(last_pos, device=device, dtype=torch.long)
            lp = logits_clean[torch.arange(B, device=device), idx, :]
            lq = logits_patched[torch.arange(B, device=device), idx, :]

            logp_clean = F.log_softmax(lp, dim=-1)
            logp_patch = F.log_softmax(lq, dim=-1)
            p_clean = logp_clean.exp()

            kl = (p_clean * (logp_clean - logp_patch)).sum(dim=-1)
            total_kl += kl.mean().item() * B

            tgt_logprob = logp_patch[torch.arange(B, device=device), target_ids]
            logit_diff  = (lq[torch.arange(B, device=device), target_ids] -
                           lq[torch.arange(B, device=device), distractor_ids])

            sum_logprob   += tgt_logprob.mean().item() * B
            sum_logitdiff += logit_diff.mean().item() * B

            if yy_ids.numel() > 0:
                p_two = logp_patch[:, yy_ids].exp()
                for j, rec in enumerate(recs_batch):
                    y0 = int(rec.get("yy_start", 0))
                    gt_mask = (yy_vals > y0); le_mask = (yy_vals <= y0)
                    if gt_mask.any() or le_mask.any():
                        pd_val = (p_two[j, gt_mask].sum() - p_two[j, le_mask].sum()).item()
                        sum_pd += pd_val; count_pd += 1

            total_B += B

            del logits_clean, logits_patched, lp, lq, logp_clean, logp_patch, p_clean, kl, tgt_logprob, logit_diff
            if device.type == "cuda":
                torch.cuda.empty_cache()

        neg_kl = - (total_kl / total_B)
        avg_logprob = (sum_logprob / total_B)
        mean_logit_diff = (sum_logitdiff / total_B)
        mean_pd = (sum_pd / max(1, count_pd)) if count_pd > 0 else None
        return {
            "neg_kl": neg_kl,
            "kl": -neg_kl,
            "mean_logprob_target": avg_logprob,
            "cross_entropy": -avg_logprob,
            "mean_logit_diff": mean_logit_diff,
            "mean_prob_diff": mean_pd,
            "n_examples": total_B,
            "two_digit_vocab_size": len(yy_to_id),
        }

    n_layers = model.cfg.n_layers
    all_clean_keep     = {L: set(range(H + (1 if include_mlp else 0))) for L in range(n_layers)}
    all_corrupted_keep = {L: set() for L in range(n_layers)}

    return {
        "subgraph": _core(keep_sets),
        "all_clean": _core(all_clean_keep),
        "all_corrupted": _core(all_corrupted_keep),
    }


# ============================ EVALUATION (IOI) ================================
@torch.no_grad()
def evaluate_ioi(
    model: HookedTransformer,
    tok,
    clean_recs: List[Dict[str, Any]],
    nodes_root: Path,
    keep_sets: Dict[int, Set[int]],
    H: int,
    include_mlp: bool,
    bs: int,
    device: torch.device,
    dtype: torch.dtype,
    debug_sanity_all_clean: bool,
) -> Dict[str, Any]:
    """
    IOI evaluation that supports multi-token target/distractor strings.

    We compute:
      - KL(clean || patched) at last token (unchanged)
      - mean_logprob_target: mean logprob of the full target token sequence
        placed at the last positions (ending at lp)
      - mean_logit_diff: mean_logprob(target_seq) - mean_logprob(distractor_seq)
      - cross_entropy: -mean_logprob_target
      - pred_frac_* computed using last-token argmax only (still meaningful but optional)
    """

    def _encode_name_ids(s: str) -> List[int]:
        # Keep dataset string exactly; allow multi-token.
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"Empty tokenization for string: {s!r}")
        return ids

    def _mean_seq_logprob(logp_patch: torch.Tensor, lp_idx: torch.Tensor, seq_ids_list: List[List[int]]) -> torch.Tensor:
        """
        logp_patch: [B, T, V] log softmax over vocab for patched run
        lp_idx:     [B] last non-pad positions
        seq_ids_list: list of token-id sequences (variable length) per example

        Returns: [B] mean logprob over the sequence, aligned so the sequence ends at lp.
        Uses positions [lp-L+1 ... lp] for a length-L sequence.
        """
        B, T, V = logp_patch.shape
        out = torch.empty(B, device=logp_patch.device, dtype=logp_patch.dtype)
        for j in range(B):
            ids = seq_ids_list[j]
            L = len(ids)
            end = int(lp_idx[j].item())
            start = end - L + 1
            if start < 0:
                # This should basically never happen if prompts are long enough,
                # but keep it explicit so failures are readable.
                raise RuntimeError(
                    f"[ioi] target/distractor sequence too long for context window in batch item {j}: "
                    f"seq_len={L}, lp={end}, T={T}."
                )
            pos = torch.arange(start, end + 1, device=logp_patch.device, dtype=torch.long)  # [L]
            tok_ids = torch.tensor(ids, device=logp_patch.device, dtype=torch.long)        # [L]
            out[j] = logp_patch[j, pos, tok_ids].mean()
        return out  # [B]

    def _core(per_layer_keep: Dict[int, Set[int]], *, debug=False) -> Dict[str, Any]:
        total_kl, total_B = 0.0, 0
        sum_logprob_tgt, sum_margin = 0.0, 0.0
        preds_tgt_last = preds_dis_last = preds_other_last = 0

        n_layers = model.cfg.n_layers
        all_layers = list(range(n_layers))

        for bidx, s, e in batches(len(clean_recs), bs):
            recs_batch = clean_recs[s:e]
            prompts = [r["prompt"] for r in recs_batch]

            Bc, Tc = cache_batch_shape(nodes_root, "clean", bidx, include_mlp=include_mlp)
            if Bc != len(prompts):
                raise RuntimeError(f"[cache/eval mismatch] batch_{bidx}: cache B={Bc} vs eval B={len(prompts)}")
            tok_batch = tokenize_to_cache_len(tok, prompts, Tc)

            input_ids = tok_batch["input_ids"].to(device)
            attn_mask = tok_batch["attention_mask"].to(device)
            last_pos = last_positions(attn_mask)
            B = input_ids.size(0)

            logits_clean = tl_call(model, input_ids, attn_mask)
            hooks = build_hooks_for_layers(
                model, tok_batch, all_layers, per_layer_keep,
                nodes_root, bidx, H, include_mlp, device, dtype
            )
            logits_patched = tl_call_with_hooks(model, input_ids, hooks, attn_mask)

            idx = torch.as_tensor(last_pos, device=device, dtype=torch.long)
            lp = logits_clean[torch.arange(B, device=device), idx, :]
            lq = logits_patched[torch.arange(B, device=device), idx, :]

            if debug:
                max_abs = (lp - lq).abs().max().item()
                print(f"[sanity all-clean] max|clean - patched| = {max_abs:.3e}")

            # KL(clean || patched) at last token (unchanged)
            logp_clean = F.log_softmax(lp, dim=-1)
            logp_patch_last = F.log_softmax(lq, dim=-1)
            p_clean = logp_clean.exp()
            kl = (p_clean * (logp_clean - logp_patch_last)).sum(dim=-1)
            total_kl += kl.mean().item() * B

            # Multi-token target/distractor logprobs aligned to end at lp.
            logp_patch_full = F.log_softmax(logits_patched, dim=-1)  # [B,T,V]
            lp_idx = idx  # [B]

            tgt_ids_list = [_encode_name_ids(r["target_str"]) for r in recs_batch]
            dis_ids_list = [_encode_name_ids(r["distractor_str"]) for r in recs_batch]

            tgt_lp = _mean_seq_logprob(logp_patch_full, lp_idx, tgt_ids_list)  # [B]
            dis_lp = _mean_seq_logprob(logp_patch_full, lp_idx, dis_ids_list)  # [B]
            margin = tgt_lp - dis_lp

            sum_logprob_tgt += tgt_lp.mean().item() * B
            sum_margin      += margin.mean().item() * B

            # Optional: prediction fractions using last-token decision only.
            # Use the *last token id* of each sequence.
            tgt_last_ids = torch.tensor([ids[-1] for ids in tgt_ids_list], device=device, dtype=torch.long)
            dis_last_ids = torch.tensor([ids[-1] for ids in dis_ids_list], device=device, dtype=torch.long)

            preds = lq.argmax(dim=-1)
            preds_tgt_last += int((preds == tgt_last_ids).sum().item())
            preds_dis_last += int((preds == dis_last_ids).sum().item())
            preds_other_last += int(B - ((preds == tgt_last_ids) | (preds == dis_last_ids)).sum().item())

            total_B += B

            del logits_clean, logits_patched, lp, lq, logp_clean, logp_patch_last, p_clean, kl, logp_patch_full, tgt_lp, dis_lp, margin
            if device.type == "cuda":
                torch.cuda.empty_cache()

        neg_kl = - (total_kl / total_B)
        mean_logprob_tgt = (sum_logprob_tgt / total_B)
        mean_margin = (sum_margin / total_B)
        return {
            "neg_kl": neg_kl,
            "kl": -neg_kl,
            "mean_logprob_target": mean_logprob_tgt,
            "cross_entropy": -mean_logprob_tgt,
            "mean_logit_diff": mean_margin,  # stored under same key for reporting
            "pred_frac_IO": preds_tgt_last / max(1, total_B),
            "pred_frac_S": preds_dis_last / max(1, total_B),
            "pred_frac_OTHER": preds_other_last / max(1, total_B),
            "n_examples": total_B,
            "ioi_metric_note": "mean_logit_diff is mean_logprob(target_seq)-mean_logprob(distractor_seq) for multi-token support; pred_fracs use last-token ids.",
        }

    n_layers = model.cfg.n_layers
    all_clean_keep     = {L: set(range(H + (1 if include_mlp else 0))) for L in range(n_layers)}
    all_corrupted_keep = {L: set() for L in range(n_layers)}

    return {
        "subgraph": _core(keep_sets, debug=False),
        "all_clean": _core(all_clean_keep, debug=debug_sanity_all_clean),
        "all_corrupted": _core(all_corrupted_keep, debug=False),
    }


# ==================== DOCSTRING EVALUATION (ONLY ADD METRICS) =================
@torch.no_grad()
def evaluate_docstring_add_metrics(
    model: HookedTransformer,
    tok,
    clean_recs: List[Dict[str, Any]],
    nodes_root: Path,
    keep_sets: Dict[int, Set[int]],
    H: int,
    include_mlp: bool,
    bs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """
    Adds:
      - mean_logprob_target
      - cross_entropy
      - mean_logit_diff

    Docstring jsonl fields:
      correct_token_ids: [id]
      wrong_token_ids:   [id...]
    distractor = hardest wrong: argmax logit among wrong_token_ids.
    """
    n_layers = model.cfg.n_layers
    all_layers = list(range(n_layers))

    def _core(per_layer_keep: Dict[int, Set[int]]) -> Dict[str, Any]:
        total_kl, total_B = 0.0, 0
        sum_logprob, sum_logitdiff = 0.0, 0.0

        for bidx, s, e in batches(len(clean_recs), bs):
            recs_batch = clean_recs[s:e]
            prompts = [r["prompt"] for r in recs_batch]

            Bc, Tc = cache_batch_shape(nodes_root, "clean", bidx, include_mlp=include_mlp)
            if Bc != len(prompts):
                raise RuntimeError(f"[cache/eval mismatch] batch_{bidx}: cache B={Bc} vs eval B={len(prompts)}")
            tok_batch = tokenize_to_cache_len(tok, prompts, Tc)

            input_ids = tok_batch["input_ids"].to(device)
            attn_mask = tok_batch["attention_mask"].to(device)
            last_pos = last_positions(attn_mask)
            B = input_ids.size(0)

            tgt_ids = torch.tensor(
                [int(r["correct_token_ids"][0]) for r in recs_batch],
                dtype=torch.long,
                device=device,
            )
            wrong_ids_list = [
                torch.tensor([int(x) for x in r["wrong_token_ids"]], dtype=torch.long, device=device)
                for r in recs_batch
            ]

            logits_clean = tl_call(model, input_ids, attn_mask)

            hooks = build_hooks_for_layers(
                model, tok_batch, all_layers, per_layer_keep,
                nodes_root, bidx, H, include_mlp, device, dtype
            )
            logits_patched = tl_call_with_hooks(model, input_ids, hooks, attn_mask)

            idx = torch.as_tensor(last_pos, device=device, dtype=torch.long)
            lp = logits_clean[torch.arange(B, device=device), idx, :]
            lq = logits_patched[torch.arange(B, device=device), idx, :]

            logp = F.log_softmax(lp, dim=-1)
            logq = F.log_softmax(lq, dim=-1)
            p = logp.exp()
            kl = (p * (logp - logq)).sum(dim=-1)
            total_kl += kl.mean().item() * B

            logq_full = F.log_softmax(lq, dim=-1)
            tgt_logprob = logq_full[torch.arange(B, device=device), tgt_ids]
            sum_logprob += tgt_logprob.mean().item() * B

            margins = []
            for j in range(B):
                max_wrong = lq[j, wrong_ids_list[j]].max()
                margins.append(lq[j, tgt_ids[j]] - max_wrong)
            mean_margin = torch.stack(margins).mean().item()
            sum_logitdiff += mean_margin * B

            total_B += B

            del logits_clean, logits_patched, lp, lq, logp, logq, p, kl, logq_full, tgt_logprob
            if device.type == "cuda":
                torch.cuda.empty_cache()

        neg_kl = - (total_kl / total_B)
        mean_logprob = (sum_logprob / total_B)
        mean_logit_diff = (sum_logitdiff / total_B)
        return {
            "neg_kl": neg_kl,
            "kl": -neg_kl,
            "mean_logprob_target": mean_logprob,
            "cross_entropy": -mean_logprob,
            "mean_logit_diff": mean_logit_diff,
            "n_examples": total_B,
        }

    n_layers = model.cfg.n_layers
    all_clean_keep     = {L: set(range(H + (1 if include_mlp else 0))) for L in range(n_layers)}
    all_corrupted_keep = {L: set() for L in range(n_layers)}

    return {
        "subgraph": _core(keep_sets),
        "all_clean": _core(all_clean_keep),
        "all_corrupted": _core(all_corrupted_keep),
    }
# ==============================================================================


@torch.no_grad()
def evaluate_generic_lastpos(
    model: HookedTransformer,
    tok,
    clean_recs: List[Dict[str, Any]],
    nodes_root: Path,
    keep_sets: Dict[int, Set[int]],
    H: int,
    include_mlp: bool,
    bs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    n_layers = model.cfg.n_layers
    all_layers = list(range(n_layers))

    def _core(per_layer_keep: Dict[int, Set[int]]) -> Dict[str, Any]:
        total_kl, total_B = 0.0, 0

        for bidx, s, e in batches(len(clean_recs), bs):
            recs_batch = clean_recs[s:e]
            prompts = [r["prompt"] for r in recs_batch]

            Bc, Tc = cache_batch_shape(nodes_root, "clean", bidx, include_mlp=include_mlp)
            if Bc != len(prompts):
                raise RuntimeError(f"[cache/eval mismatch] batch_{bidx}: cache B={Bc} vs eval B={len(prompts)}")
            tok_batch = tokenize_to_cache_len(tok, prompts, Tc)

            input_ids = tok_batch["input_ids"].to(device)
            attn_mask = tok_batch["attention_mask"].to(device)
            last_pos = last_positions(attn_mask)
            B = input_ids.size(0)

            logits_clean = tl_call(model, input_ids, attn_mask)

            hooks = build_hooks_for_layers(
                model, tok_batch, all_layers, per_layer_keep,
                nodes_root, bidx, H, include_mlp, device, dtype
            )
            logits_patched = tl_call_with_hooks(model, input_ids, hooks, attn_mask)

            idx = torch.as_tensor(last_pos, device=device, dtype=torch.long)
            lp = logits_clean[torch.arange(B, device=device), idx, :]
            lq = logits_patched[torch.arange(B, device=device), idx, :]

            logp = F.log_softmax(lp, dim=-1)
            logq = F.log_softmax(lq, dim=-1)
            p = logp.exp()
            kl = (p * (logp - logq)).sum(dim=-1)

            total_kl += kl.mean().item() * B
            total_B += B

            del logits_clean, logits_patched, lp, lq, logp, logq, p, kl
            if device.type == "cuda":
                torch.cuda.empty_cache()

        neg_kl = - (total_kl / total_B)
        return {"neg_kl": neg_kl, "kl": -neg_kl, "n_examples": total_B}

    n_layers = model.cfg.n_layers
    all_clean_keep     = {L: set(range(H + (1 if include_mlp else 0))) for L in range(n_layers)}
    all_corrupted_keep = {L: set() for L in range(n_layers)}

    return {
        "subgraph": _core(keep_sets),
        "all_clean": _core(all_clean_keep),
        "all_corrupted": _core(all_corrupted_keep),
    }

# ============================ RUN ONE TASK ====================================
def run_task(model: HookedTransformer, tok, task_key: str, spec: Dict[str, Any]) -> None:
    set_seed(int(spec["seed"]))

    out_tag = spec["out_tag"]
    out_dir = ensure_out_dir(out_tag)

    nodes_root = resolve_cache_nodes_root(task_key, CONFIG["MODEL_KEY"])
    if not nodes_root.exists():
        raise FileNotFoundError(f"[{task_key}] cache not found: {nodes_root}")

    # Optional cache meta check (helps catch wrong bs)
    meta = read_cache_meta(nodes_root)
    if meta is not None:
        cache_bs = int(meta.get("batch_size", spec["batch_size"]))
        if int(spec["batch_size"]) != cache_bs:
            raise RuntimeError(f"[{task_key}] batch_size mismatch: spec bs={spec['batch_size']} vs cache meta bs={cache_bs}")

    clean_recs = read_jsonl(spec["clean_jsonl"])

    H = int(spec["num_heads"])
    include_mlp = bool(spec["include_mlp"])
    bs = int(spec["batch_size"])
    dtype = torch_dtype_from_str(CONFIG["DTYPE"])
    device = torch.device(CONFIG["DEVICE"])

    total_nodes = int(spec["num_layers"]) * int(spec["nodes_per_layer"])
    gid_to_Li: List[Tuple[int, int]] = [(L, i) for L in range(int(spec["num_layers"])) for i in range(int(spec["nodes_per_layer"]))]

    t0 = time.time()
    f_full = dataset_value_global(
        model, tok, clean_recs, nodes_root,
        S_global=set(range(total_nodes)),
        gid_to_Li=gid_to_Li,
        H=H, include_mlp=include_mlp,
        bs=bs, device=device, dtype=dtype
    )
    f_empty = dataset_value_global(
        model, tok, clean_recs, nodes_root,
        S_global=set(),
        gid_to_Li=gid_to_Li,
        H=H, include_mlp=include_mlp,
        bs=bs, device=device, dtype=dtype
    )

    repeats = int(spec["repeats"])
    k_min = int(spec["k_min"])
    k_max = int(spec["k_max"])
    assert 1 <= k_min <= k_max < total_nodes

    phi_sum = torch.zeros(total_nodes, dtype=torch.float64, device=device)
    phi_cnt = torch.zeros(total_nodes, dtype=torch.int64, device=device)

    samples_path = out_dir / "global_samples.jsonl"
    with samples_path.open("w") as sf:
        for rep in range(repeats):
            K = random.randint(k_min, k_max)
            S = set(sorted(random.sample(range(total_nodes), K)))
            Mc = set(range(total_nodes)) - S

            f_S = dataset_value_global(model, tok, clean_recs, nodes_root, S,  gid_to_Li, H, include_mlp, bs, device, dtype)
            f_Mc = dataset_value_global(model, tok, clean_recs, nodes_root, Mc, gid_to_Li, H, include_mlp, bs, device, dtype)

            sv_S  = 0.5 * (f_full - f_Mc + f_S - f_empty)
            sv_Mc = 0.5 * (f_full - f_S  + f_Mc - f_empty)

            for gid in S:
                phi_sum[gid] += sv_S
                phi_cnt[gid] += 1
            for gid in Mc:
                phi_sum[gid] += sv_Mc
                phi_cnt[gid] += 1

            sf.write(json.dumps({
                "repeat": rep + 1,
                "subset_size": K,
                "f_full": f_full,
                "f_empty": f_empty,
                "f_S": f_S,
                "f_M_minus_S": f_Mc,
                "sv_group_S": sv_S,
                "sv_group_M_minus_S": sv_Mc,
            }) + "\n")

    per_node = []
    phi_all = []
    for gid in range(total_nodes):
        est = (phi_sum[gid] / max(1, int(phi_cnt[gid]))).item() if phi_cnt[gid] > 0 else 0.0
        L, i = gid_to_Li[gid]
        row = {
            "global_id": gid,
            "layer": L,
            "node_index": i,
            "node_name": describe_node(i, H),
            "phi_estimate_full_equation": est,
            "num_samples_included": int(phi_cnt[gid].item()),
        }
        per_node.append(row)
        phi_all.append(est)

    per_node_path = out_dir / "global_phi.json"
    with per_node_path.open("w") as f:
        json.dump({
            "config": spec,
            "runner_config": {"OUT_ROOT": OUT_ROOT, "OUT_GROUP": OUT_GROUP, **{k: CONFIG[k] for k in ["MODEL_KEY","TLENS_MODEL_NAME","DEVICE","DTYPE","CACHE_OUTPUT_ROOT"]}},
            "baselines": {"f_full": f_full, "f_empty": f_empty, "gap": f_full - f_empty},
            "per_node": per_node,
        }, f, indent=2)

    csv_path = out_dir / "global_phi.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_node[0].keys()))
        writer.writeheader()
        writer.writerows(per_node)

    t1 = time.time()
    runtime_phi = t1 - t0

    top_n = int(spec["top_n_per_layer"])
    pct = float(spec["global_percentile"])
    scores = torch.tensor(phi_all, dtype=torch.float32)
    thresh = torch.quantile(scores, torch.tensor(pct, dtype=torch.float32)).item()

    selected: Dict[int, Set[int]] = {L: set() for L in range(int(spec["num_layers"]))}
    layer_buckets: Dict[int, List[Tuple[int, float, int]]] = {L: [] for L in range(int(spec["num_layers"]))}
    for row in per_node:
        layer_buckets[row["layer"]].append((row["node_index"], row["phi_estimate_full_equation"], row["global_id"]))

    for L in range(int(spec["num_layers"])):
        vals_sorted = sorted(layer_buckets[L], key=lambda t: (-t[1], t[0]))
        for i, _, _ in vals_sorted[:max(0, min(top_n, len(vals_sorted)))]:
            selected[L].add(i)

    kept_percentile = []
    for gid, phi in enumerate(phi_all):
        if phi >= thresh:
            L, i = gid_to_Li[gid]
            if i not in selected[L]:
                kept_percentile.append({"global_id": gid, "layer": L, "node_index": i, "phi": float(phi)})
            selected[L].add(i)

    selection_path = out_dir / "selected_subgraph.json"
    with selection_path.open("w") as f:
        json.dump({
            "config": spec,
            "selection": {str(L): sorted(list(nodes)) for L, nodes in selected.items()},
            "top_n_per_layer": top_n,
            "global_percentile_threshold": pct,
            "percentile_value": thresh,
            "kept_by_percentile": kept_percentile,
        }, f, indent=2)

    t_eval0 = time.time()
    if spec["task_type"] == "gt":
        eval_metrics = evaluate_gt(model, tok, clean_recs, nodes_root, selected, H, include_mlp, bs, device, dtype)
    elif spec["task_type"] == "ioi":
        eval_metrics = evaluate_ioi(
            model, tok, clean_recs, nodes_root, selected, H, include_mlp, bs, device, dtype,
            debug_sanity_all_clean=bool(spec.get("debug_sanity_all_clean", False)),
        )
    elif spec["task_type"] == "docstring":
        eval_metrics = evaluate_docstring_add_metrics(
            model, tok, clean_recs, nodes_root, selected, H, include_mlp, bs, device, dtype
        )
    elif spec["task_type"] == "generic_lastpos":
        eval_metrics = evaluate_generic_lastpos(model, tok, clean_recs, nodes_root, selected, H, include_mlp, bs, device, dtype)
    else:
        raise ValueError(f"Unknown task_type: {spec['task_type']}")
    t_eval1 = time.time()

    final_path = out_dir / "global_subgraph_summary.json"
    with final_path.open("w") as f:
        json.dump({
            "config": spec,
            "runner_config": {"OUT_ROOT": OUT_ROOT, "OUT_GROUP": OUT_GROUP, **{k: CONFIG[k] for k in ["MODEL_KEY","TLENS_MODEL_NAME","DEVICE","DTYPE","CACHE_OUTPUT_ROOT"]}},
            "baselines": {"f_full": f_full, "f_empty": f_empty, "gap": f_full - f_empty},
            "runtimes_seconds": {"phi_estimation": runtime_phi, "evaluation": t_eval1 - t_eval0},
            "selection": {str(L): sorted(list(nodes)) for L, nodes in selected.items()},
            "evaluation": eval_metrics,
        }, f, indent=2)

    print(f"\n[{task_key}] wrote -> {out_dir.resolve()}")
    print(f"[{task_key}] eval(subgraph):", eval_metrics.get("subgraph"))


# ============================ PROGRAM ENTRY ===================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        nargs="+",
        default=["all"],
        help="Tasks to run: all, gt, ioi, docstring (you can pass multiple: --task gt ioi).",
    )
    return p.parse_args()

def main():
    args = parse_args()

    if len(args.task) == 1 and args.task[0].lower() == "all":
        task_keys = list(CONFIG["TASKS"].keys())
    else:
        task_keys = [t.lower().strip() for t in args.task]

    for t in task_keys:
        if t not in CONFIG["TASKS"]:
            raise ValueError(f"Unknown task: {t}. Known: {sorted(CONFIG['TASKS'].keys())} or 'all'.")

    device = torch.device(CONFIG["DEVICE"])
    print(f"[load] TLens='{CONFIG['TLENS_MODEL_NAME']}' device='{device}' model_key='{CONFIG['MODEL_KEY']}'")
    model = HookedTransformer.from_pretrained(CONFIG["TLENS_MODEL_NAME"], device=str(device))
    model.eval()

    tok = model.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        print(f"[model cfg] n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads} d_model={model.cfg.d_model}")
    except Exception:
        pass

    for task_key in task_keys:
        print(f"\n=== RUN TASK '{task_key}' ===")
        run_task(model, tok, task_key, CONFIG["TASKS"][task_key])

    print("\n[ALL DONE] ran tasks:", task_keys)
    print(f"[outputs] root = {(Path(OUT_ROOT) / OUT_GROUP).resolve()}")

if __name__ == "__main__":
    main()