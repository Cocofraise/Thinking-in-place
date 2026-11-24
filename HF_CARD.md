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

This repository contains code and a paper, not weights. The patch modifies the inference graph so that a frozen, pretrained model re-applies its own full-attention blocks several times per generated token, each time pulled back toward the block's original input. You keep your GGUF and set one variable.

- **Author:** Vincent Mathieu Malet · ORCID [0009-0003-9945-7296](https://orcid.org/0009-0003-9945-7296) · independent researcher
- **Paper:** *Thinking in Place: Anchored Recurrent Depth and the Convergence of Modeling*, in `paper/` (English and French)
- **Base:** llama.cpp `af5c583` (2026-11-23)
- **Licence:** non-commercial. Thinking in Place additions under PolyForm Noncommercial 1.0.0, documentation and paper under CC BY-NC 4.0, llama.cpp itself under MIT. **Commercial use requires a licence from the author** (see `LICENSE-TIP.md`).

## The mechanism in four lines

```
s_0     = block input                         (the anchor)
in_t    = (1 - beta) * s_t + beta * s_0       (t > 0)
s_{t+1} = Block(in_t)                         (attention + FFN)
R(l) = max(1, round(N·√(l/l_max)))            beta = 1 - φ⁻¹ / N
```

Only full-attention layers loop. Recurrent (gated-delta-net) layers run once. Only predicting passes loop; prompt processing is single-pass.

## Use

Apply `patch/thinking-in-place-vs-llama.cpp-af5c583.diff` to llama.cpp at commit `af5c583`, build as usual, then:

```bash
TIP_N=8 ./llama-server -m model.gguf -c 8192 -ngl 99
```

Unset `TIP_N` and you get stock llama.cpp, with identical output. Every setting is documented in `TIP.md`.

## What it buys

The loop helps where a single pass is fragile, and only there. Nine reasoning items with a checkable answer, six variants each, temperature 0, on a 9B model at its shipped quantization (Q5) and one step lower (Q4):

| Model | Single pass | `TIP_N=8` |
|---|---|---|
| Q5 | 47/54 | 50/54 |
| Q4 | 45/54 | 53/54 |
| Both | 92/108 | 103/108 |

Paired item by item, the loop wins 13 answers and loses 2 (exact two-sided p = 0.007). Seven of the nine items are at ceiling in both settings; the gain comes from the two items the single pass gets wrong most often. Two quantization steps lower, where the model degenerates, the loop recovers nothing: it reaches a capacity that is there, it does not replace one that is gone.

On public benchmarks, same model: GSM8K (first 200 test questions) is saturated, 195 against 192, nothing to recover; on MATH-500 levels 4–5 with a numeric answer (158 questions) the loop goes from 124 (78.5%) to 134 (84.8%), 19 answers won against 9 lost (p = 0.087). No gain where the model is already right, a gain where it is not.

## What it supports

| Area | Status |
|---|---|
| Models | Qwen3.5-style hybrids in llama.cpp: dense (9B, 27B) and mixture-of-experts (35B) |
| Hardware | GPU, CPU-only, MoE experts in host memory (`--n-cpu-moe`), MoE expert cache (`--moe-cache-mib`) |
| Speculative decoding | `draft-mtp` and `draft-dflash` return the same text as the loop alone, 2.3 to 2.6 times faster |
| Parallel slots | 4 and 8 slots, dense and MoE |
| MoE routing under the loop | two modes: frozen at the first iteration (default) or re-decided at every iteration |

## Cost

At about 13,500 tokens of context, without speculative decoding:

| Model | Single pass | `TIP_N=8` |
|---|---|---|
| 9B | 16.2 ms/token | 34.4 ms/token (×2.1) |
| 35B MoE, experts of 16 layers on CPU | 18.5 ms/token | 31.4 ms/token (×1.7) |

Time per token is linear in the number of block evaluations. With `draft-mtp` the 9B at `TIP_N=8` runs at 81 tokens/s, faster than the single pass without speculation (59 tokens/s).

## Evidence

| Claim | Basis |
|---|---|
| Loop off is identical to unpatched llama.cpp | same text and first-token log-probabilities, dense and MoE |
| The loop executes exactly the scheduled iterations | 0.49 ms per block evaluation across three budgets |
| Sampling temperature is not amplified | 3,024 positions replayed along identical prefixes; same gap at T = 0, 0.7 and 1.0 |
| No repetition collapse in long generations | 1,400-token generations, T = 0, 0.7 and 1.0 |
| The loop does not move a quantized model's probabilities closer to the original | same distance with and without the loop, dense and MoE |
| sqrt schedule at budget 8 ≈ uniform budget 12, at about half the loop compute | paper, internal blind panels, small n |
| Iterate settles in about three iterations | paper, per-layer probes |
| Reasoning benefit when the model actually deliberates | paper, two models, a handful of items |
| Frozen and free MoE routing behave differently | paper, full outputs read side by side, one run per mode and prompt |

The quality results are internal evaluations on one model family and quantized checkpoints; holdout replication is pending. None of them is a public benchmark.

## Intended use

- Trading inference time for a more settled reading on existing checkpoints, without touching the weights.
- Research on recurrent depth as a pure inference-time operator: schedule, anchor strength, convergence, expert routing under iteration.

## Limits

- The loop amplifies what the base configuration does, good or bad. It does not repair an unstable setup.
- Too many iterations make results worse.
- No benefit on tasks whose difficulty lies in reading the prompt.
- One model family.

## How this was made

Research, code and writing were carried out with AI assistance. The author designed the mechanism and directed the experiments, and has read the results, the generated outputs and the conclusions, for which he takes responsibility. The prose was drafted with AI assistance and has not been reviewed word by word by the author.

## How this differs from trained looped models

Trained looped models learn a block that is meant to be re-applied. Thinking in Place needs no training: it works on checkpoints that already exist, block by block, with an anchor and a depth schedule. The two can be combined.
