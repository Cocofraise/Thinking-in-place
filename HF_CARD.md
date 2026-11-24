---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0/
library_name: llama.cpp
tags:
  - llama.cpp
  - inference
  - recurrent-depth
  - looped-transformer
  - training-free
  - mixture-of-experts
language:
  - en
---

# Thinking in Place (TiP)

**Training-free recurrent depth for hybrid attention / state-space models, as a llama.cpp patch.**

This repository contains code, not weights. It modifies the inference graph so that a frozen, pretrained model re-applies its own full-attention blocks several times per generated token, each time pulled back toward the block's original input.

- **Author:** Vincent Mathieu Malet · ORCID [0009-0003-9945-7296](https://orcid.org/0009-0003-9945-7296) · independent researcher
- **Paper:** *Thinking in Place: Anchored Recurrent Depth and the Convergence of Modeling* (draft)
- **Base:** llama.cpp `af5c583` (2026-11-23)
- **Licence:** non-commercial. Thinking in Place additions under PolyForm Noncommercial 1.0.0, documentation and paper under CC BY-NC 4.0, llama.cpp itself under MIT. **Commercial use requires a licence from the author** (see `LICENSE-TIP.md`).

## The mechanism in four lines

```
s_0     = block input                         (the anchor)
in_t    = (1 - beta) * s_t + beta * s_0       (t > 0)
s_{t+1} = Block(in_t)                         (attention + FFN)
R(layer) grows with depth: sqrt ramp, full budget on the deepest attention layer
```

Only full-attention layers loop. Recurrent (gated-delta-net) layers run once. Only predicting passes loop; prompt processing is single-pass.

## Use

```bash
TIP_N=8 TIP_DEPTH_LAW=sqrt TIP_BETA=0.9485 \
  ./llama-server -m model.gguf -c 8192 -ngl 99
```

Unset `TIP_N` and you get stock llama.cpp. Full list of settings: see `TIP.md`.

## Intended use

- Trading inference time for a more settled reading on models of the Qwen3.5-style hybrid family (dense and mixture-of-experts), without touching the weights.
- Research on recurrent depth as a pure inference-time operator: schedule, anchor strength, convergence, expert routing under iteration.

## Evidence and its limits

| Claim | Status |
|---|---|
| Loop off is identical to unpatched llama.cpp | Checked on this port (9B dense, 35B MoE): identical text and first-token log-probabilities |
| Loop on changes the distribution and stays coherent | Checked on this port, two prompts per model |
| Loop on with speculative decoding (`draft-mtp`, `draft-dflash`) gives the same text, 2.3 to 2.6 times faster | Checked on this port, 9B, two prompts |
| Sampling temperature is not amplified by the loop at the validated setting | Checked on this port, 9B, 3,024 replayed positions and 28 long generations |
| sqrt schedule at budget 8 ≈ uniform budget 12, about 45% less loop compute | Paper, internal blind panels, small n |
| Iterate settles in about three iterations | Paper, per-layer probes |
| Fewer repetition collapses in long generations | Paper, internal workloads |
| Reasoning benefit when the model actually deliberates | Paper, two models, a handful of items |

Everything in the four rows marked "Paper" is an internal evaluation on one model family and quantized checkpoints. Holdout replication is pending. None of it is a public benchmark.

## Cost

About ×1.5 to ×1.65 on a 9B hybrid, ×2.7 on a 35B MoE hybrid, ×3.4 on a 27B dense model, at `R=8` with the sqrt schedule and a context of about 11k tokens (author's measurements on an earlier research fork).

## Known limitations

- The loop amplifies what the base configuration does, good or bad. It does not repair an unstable setup.
- Too many iterations make results worse.
- No benefit on tasks whose difficulty lies in reading the prompt.
- Tested with a single slot only.
- Several options are ported but not yet exercised on this port; the convergence probe and the CUDA expert-gather optimisation are not ported.

## How this differs from trained looped models

Trained looped models learn a block that is meant to be re-applied. Thinking in Place needs no training: it works on checkpoints that already exist, block by block, with a fixed anchor and a depth schedule. The two can be combined.
