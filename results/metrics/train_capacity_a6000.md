# Fine-tuning capacity measurement — RTX A6000 (48 GiB)

Input for the pending fine-tuning batch-size decision. Not part of the
zero-shot benchmark; kept here because the numbers are hardware-specific and
would otherwise be re-derived by hand.

## Method

Synthetic full fine-tune step: `clip.load(...).float().train()`, AdamW
(lr=1e-5), bf16 autocast with fp32 master weights, random image/token batches at
the model's native resolution, 5 steps, median of the last 3, CUDA-synchronised.
Peak is `torch.cuda.max_memory_allocated()`. Gradient checkpointing wraps every
`resblocks` entry of both towers (`use_reentrant=False`).

Epoch estimate is 34,054 train images / batch × step time — the Q11 definition
(1 epoch = one pass over images, one of the two captions sampled per image).

## Measured

| model | batch | ckpt | peak VRAM | step | epoch |
|---|---|---|---|---|---|
| ViT-L/14 | 32 | off | 14.95 GiB | 396 ms | 7.0 min |
| ViT-L/14 | 64 | off | 24.32 GiB | 704 ms | 6.2 min |
| ViT-L/14 | 128 | off | 43.11 GiB | 1307 ms | 5.8 min |
| ViT-L/14 | 256 | off | OOM | | |
| ViT-L/14 | 32 | on | 8.03 GiB | 483 ms | 8.6 min |
| ViT-L/14 | 64 | on | 8.05 GiB | 873 ms | 7.7 min |
| ViT-L/14 | 128 | on | 9.01 GiB | 1634 ms | 7.2 min |
| ViT-L/14 | 256 | on | 12.68 GiB | 3185 ms | 7.1 min |
| ViT-L/14 | 512 | on | 20.06 GiB | 6320 ms | 7.0 min |
| ViT-B/16 | 128 | off | 14.40 GiB | 395 ms | 1.8 min |
| ViT-B/16 | 256 | off | 26.83 GiB | 747 ms | 1.7 min |
| ViT-B/16 | 512 | off | OOM | | |
| ViT-B/16 | 256 | on | 5.43 GiB | 938 ms | 2.1 min |
| ViT-B/16 | 512 | on | 8.90 GiB | 1829 ms | 2.0 min |
| ViT-B/32 | 256 | off | 12.55 GiB | 325 ms | 0.7 min |
| ViT-B/32 | 512 | off | 22.99 GiB | 614 ms | 0.7 min |
| ViT-B/32 | 256 | on | 3.26 GiB | 402 ms | 0.9 min |
| ViT-B/32 | 512 | on | 4.75 GiB | 767 ms | 0.9 min |

Full sweep (batches 32–512 × ckpt on/off × three models) is the source of the
above; rows omitted here are strictly dominated.

## What it settles

The earlier 16 GiB machine capped ViT-L/14 at batch 16, which with a PK sampler
at K=2 left **8 identities per batch** — too few for a contrastive loss to say
anything about model scale. That constraint is gone.

- Largest batch common to all three models **without** checkpointing is 128,
  but ViT-L/14 sits at 43.11 GiB of 49.1 GiB there — no headroom for the
  allocator, the val pass, or a longer caption batch.
- **With** checkpointing, batch 256 costs ViT-L/14 12.68 GiB and *5% more time
  per epoch than batch 128 without it* (7.1 vs 5.8 min) — recomputation is
  almost entirely paid back by batch efficiency.

Recomputation cost falls as batch grows because the per-step overhead amortises:
ViT-L/14 epoch time is 8.6 min at batch 32 but 7.0 min at batch 512.
