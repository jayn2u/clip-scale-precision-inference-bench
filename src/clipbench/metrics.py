"""Retrieval metrics and fp32-reference drift metrics.

Retrieval follows the CUHK-PEDES protocol: every caption is a query against the
full 3,074-image gallery, and a hit is a person-`id` match, so other images of
the same identity count. R@K reports whether the user would see a correct
result; mAP is the more sensitive instrument, since identities carry several
gallery images and mAP moves when ranks shift below the top-K cutoff.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def similarity_matrix(text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
    """Cosine similarity, shape (n_captions, n_images)."""
    return l2_normalize(text_features.astype(np.float64)).astype(np.float32) @ l2_normalize(
        image_features.astype(np.float64)
    ).astype(np.float32).T


def retrieval_metrics(
    sim: np.ndarray,
    caption_ids: np.ndarray,
    image_ids: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    order = np.argsort(-sim, axis=1, kind="stable")
    matches = image_ids[order] == caption_ids[:, None]

    out = {f"R@{k}": float(matches[:, :k].any(axis=1).mean() * 100) for k in ks}

    cumulative = np.cumsum(matches, axis=1, dtype=np.int32)
    ranks = np.arange(1, matches.shape[1] + 1, dtype=np.float32)
    precision_at_hit = (cumulative / ranks) * matches
    n_positives = matches.sum(axis=1)
    # Every caption's identity is in the gallery by construction, but guard
    # anyway so a subset run cannot divide by zero.
    valid = n_positives > 0
    ap = np.zeros(matches.shape[0], dtype=np.float64)
    ap[valid] = precision_at_hit[valid].sum(axis=1) / n_positives[valid]
    out["mAP"] = float(ap[valid].mean() * 100)
    return out


def embedding_drift(reference: np.ndarray, current: np.ndarray) -> dict:
    """Per-vector cosine similarity against the fp32 reference embeddings."""
    ref = l2_normalize(reference.astype(np.float64))
    cur = l2_normalize(current.astype(np.float64))
    cos = (ref * cur).sum(axis=1)
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    percentiles = {f"p{q}": float(v) for q, v in zip(qs, np.percentile(cos, qs))}
    return {
        "cosine_mean": float(cos.mean()),
        "cosine_min": float(cos.min()),
        "cosine_p1": percentiles["p1"],
        "cosine_std": float(cos.std()),
        "cosine_percentiles": percentiles,
    }


def ranking_agreement(
    sim_reference: np.ndarray,
    sim_current: np.ndarray,
    tau_sample: int = 500,
    seed: int = 0,
) -> dict:
    """Did the numerical drift actually change what the user gets back?

    Cosine similarity answers "how far did the embedding move"; this answers
    "did the ranking move". The gap between the two is the interesting result.
    """
    from scipy.stats import kendalltau

    order_ref = np.argsort(-sim_reference, axis=1, kind="stable")
    order_cur = np.argsort(-sim_current, axis=1, kind="stable")

    top1 = float((order_ref[:, 0] == order_cur[:, 0]).mean() * 100)

    k = min(10, order_ref.shape[1])
    overlap = [
        len(set(order_ref[i, :k].tolist()) & set(order_cur[i, :k].tolist()))
        for i in range(order_ref.shape[0])
    ]
    top10 = float(np.mean(overlap) / k * 100)

    rng = np.random.default_rng(seed)
    n_queries = sim_reference.shape[0]
    sample = rng.choice(n_queries, size=min(tau_sample, n_queries), replace=False)
    taus = [kendalltau(sim_reference[i], sim_current[i]).statistic for i in sample]

    return {
        "top1_agreement_pct": top1,
        "top10_overlap_pct": top10,
        "kendall_tau_mean": float(np.mean(taus)),
        "kendall_tau_min": float(np.min(taus)),
        "kendall_tau_n_queries": int(len(sample)),
    }
