# Thinking in Place (TiP) — training-free recurrent depth for llama.cpp

A small patch on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) that lets a **frozen, pretrained** hybrid model re-apply its own attention blocks several times per generated token. No fine-tuning, no adapters, no extra weights: you load the same GGUF and set an environment variable.

Paper: *Thinking in Place: Anchored Recurrent Depth and the Convergence of Modeling* (draft, Vincent Mathieu Malet, ORCID [0009-0003-9945-7296](https://orcid.org/0009-0003-9945-7296)).

## What it does

Each selected full-attention block (norm → attention → residual → norm → FFN → residual) is iterated `R` times on its own hidden state. Every iteration after the first is pulled back toward the block's original input, the **anchor**:

```
s_0     = block input
in_t    = (1 - beta) * s_t + beta * s_0      (t > 0)
s_{t+1} = Block(in_t)
```

Four rules keep this safe on a model that was never trained to be iterated:

1. **Anchor every iteration.** Without the blend the iterate drifts out of the region the layer knows. With `beta ≈ 0.947` each step is mostly anchor plus a controlled share of the refined state.
2. **Spend depth where it matters.** Iterations are allocated per layer by a law of depth. The square-root ramp gives the deepest attention layer the full budget `R` and shallow layers close to one pass.
3. **Loop only where a token is predicted.** Decode steps and small verification batches (≤ 32 tokens per sequence) are looped. Prompt processing runs single-pass.
4. **Never loop a recurrent layer.** On hybrid attention / state-space models the gated-delta-net layers carry a sequential state that must advance exactly once per token. They always run once.

K/V entries of the current positions are rewritten at each iteration; the last iteration is what later tokens attend to.

## Supported models

The Qwen3.5-style hybrid family as implemented by llama.cpp:

| File | Models |
|---|---|
| `src/models/qwen35.cpp` | dense hybrid (e.g. 9B, 27B) |
| `src/models/qwen35moe.cpp` | mixture-of-experts hybrid (e.g. 35B) |

All the logic and every knob live in `src/models/tip.h`. The patch also touches four shared files of llama.cpp, about forty lines marked `TIP`, to let the MoE expert cache serve several selections of one layer in the same pass: `src/llama-moe-cache.cpp`, `src/llama-moe-cache.h`, `src/llama-graph.cpp`, `src/llama-graph.h`.

## Quick start

Build llama.cpp as usual, then:

```bash
TIP_N=8 TIP_DEPTH_LAW=sqrt TIP_BETA=0.9485 \
  ./llama-server -m model.gguf -c 8192 -ngl 99
```

With `TIP_N` unset (or `1`) the graph is the stock llama.cpp graph.

## Settings

| Variable | Meaning | Default |
|---|---|---|
| `TIP_N` | `R`, iterations of the deepest looped layer | `1` (off) |
| `TIP_BETA` | anchor strength in `[0,1]` | `0.5` |
| `TIP_DEPTH_LAW` | `sqrt`, `log` or `square`: taper `R` with layer index | flat |
| `TIP_LAYERS` | comma-separated layer indices to loop | all full-attention layers |
| `TIP_BETA_LAW` | `phi`: per-layer `beta = 1 - 0.618 / R(layer)` | off |
| `TIP_BETA_SCHEDULE` | `linear`: anneal `beta` from `BETA_MIN` to `beta` across iterations | off |
| `TIP_BETA_MIN` | start of the anneal | `0.3` |
| `TIP_FFN_LAST` | `1`: run the FFN only on the last iteration (changes the fixed point) | `0` |
| `TIP_FREEZE_ROUTE` | MoE only: keep the experts chosen at iteration 0 for the whole loop | `1` |
| `TIP_DECODE_MAX` | largest batch (tokens per sequence) still looped | `32` |
| `TIP_PREFILL` | `1`: loop prompt-processing batches too | `0` |
| `TIP_ADAPTIVE_TAU` | per-token freeze once `cos(s_t, s_{t-1}) ≥ tau` | `0` (off) |
| `TIP_ADAPTIVE_SOFT` | width of a soft (sigmoid) freeze instead of a hard one | `0` |

The configuration the paper validates is `N=8`, `DEPTH_LAW=sqrt`, `BETA=0.9485`, full block looped.

### Expert routing inside the loop (MoE)

The expert router sits inside the looped block. This gives two distinct modes, chosen for what they do to the output:

- **Frozen (default).** The experts are chosen once, at the first iteration, and every later iteration of that token uses the same set. Only the discrete choice is frozen; the mixing weights are recomputed each iteration. It is the same reader re-reading: the mode for catching a specific error.
- **Free (`TIP_FREEZE_ROUTE=0`).** The router picks again at every iteration. The reader changes each time: the mode for the richness of open-ended generation.

On a small panel of Bayesian trap questions (35B, reasoning enabled, n = 4 items) the frozen mode caught its own errors on the two items where the two modes differed. Treat this as a signal, not a law.

Both modes also work with the MoE expert cache of llama.cpp (`--moe-cache-mib`), which keeps the experts in host memory and uploads the selected ones to the GPU. That cache originally handled one selection per layer and per pass; this patch lets a layer present several, so that free routing uploads the experts of each iteration.

Speed: without the cache the two modes run at the same speed (within about 1%). With the cache, frozen routing is about 8% faster (27.8 against 25.7 tokens/s on the 35B, experts of 16 layers in host memory, 1 GiB cache), because the experts of a token are uploaded once instead of once per iteration.

## Cost

The loop adds compute only on full-attention layers and only on predicting passes. Measured overhead at `R=8`, sqrt law, real context depth of about 11k tokens, no speculative decoding (author's measurements, earlier research fork):

| Model | Slowdown |
|---|---|
| 9B hybrid | ×1.5 to ×1.65 |
| 35B hybrid MoE | ×2.7 |
| 27B dense | ×3.4 |

The cost grows with the share of the model that is looped and with the weight of the FFN.

Measured on this port at about 13,500 tokens of context, no speculation: 9B 16.2 → 34.4 ms per token (×2.1); 35B with the experts of 16 layers on CPU 18.5 → 31.4 ms (×1.7). On the 9B the time per token is linear in the number of block evaluations (0.49 ms each), so a uniform budget of 12 costs about ×3.7.

## What has been checked on this port

Upstream base: llama.cpp `af5c583` (2026-11-23). CUDA build, temperature 0.

| Model | Loop off vs unpatched llama.cpp | Loop on (`R=8`, sqrt, `beta=0.9485`) |
|---|---|---|
| 9B dense hybrid | identical text and identical first-token log-probabilities | probabilities change, output coherent, ×1.7 slower on a short context |
| 35B MoE hybrid | identical text and identical first-token log-probabilities | probabilities change, output coherent; frozen and free routing give different probabilities |

Speculative decoding, loop on (`R=8`, sqrt), 9B hybrid, same two prompts:

| Speculation | Text vs loop without speculation | Speed vs loop without speculation | Draft acceptance |
|---|---|---|---|
| `--spec-type draft-mtp` | identical | ×2.6 | 80% |
| `--spec-type draft-dflash` (two drafters tried) | identical | ×2.3 | 74% |

Verification batches are short, so they go through the looped path. `draft-dflash2`, `draft-dspark`, `draft-eagle3` and the n-gram drafters have not been tried.

This is a wiring check on two prompts per configuration. It shows the patch is inert when off and active when on. It is **not** a quality evaluation.

## Results reported in the paper

Internal evaluations, small n, one model family, quantized checkpoints, consumer hardware. Holdout replication is pending.

- The square-root schedule at budget 8 matched a uniform budget of 12 on blind panels at about half the loop compute (53% fewer block iterations). A uniform budget of 16 was worse than both.
- Per-layer probes show the iterate settling within about three iterations.
- Long unguarded generations collapsed into repetition far less often under the anchored loop.
- The reasoning benefit appears when the model actually deliberates. A model that leaves its reasoning block empty is neither helped nor harmed. Forcing reasoning off on a model trained to deliberate degrades the single pass, and the loop amplifies that.

## Limits

- The loop is an **amplifier, not a corrector**: it sharpens what the underlying configuration already does, including its instabilities.
- More iterations are not monotonically better. Past the peak, quality regresses.
- Prompt processing is not looped, so tasks whose difficulty lies in reading a long input get no benefit.
- One model family. Pure-attention stacks have no state-space majority to bound the cost.
- Not ported from the research fork: the convergence probe, per-request control of `R`, and the CUDA expert-gather kernel that recovers part of the MoE cost.
- `TIP_BETA_LAW`, `BETA_SCHEDULE`, `FFN_LAST` and the adaptive freeze are ported but have not been exercised on this port.
- Concurrency above one slot has not been tested on this port.

## Related work

Looped and recurrent-depth transformers are an active line of work, mostly **trained** with the loop: the model is pretrained or fine-tuned so that a shared block can be applied several times. SenseNova-Loop-9B (November 2026) is a recent example on a model of similar shape: a block of 24 layers is passed twice, with learned two-stream residual mixing.

This patch takes the other road. Nothing is trained. It iterates individual attention blocks of an existing checkpoint, with a fixed anchor, a depth schedule, and a rule for which passes are looped. The two approaches are not exclusive.

Direction sparked by a survey of loop-transformer papers (arXiv 2604.07822, 2604.11791, 2604.12946, 2507.10524). The anchoring, the schedule and the port are the author's own.

## Licence

Shared for non-commercial use.

- llama.cpp itself stays MIT (`LICENSE`).
- The Thinking in Place additions are under the PolyForm Noncommercial License 1.0.0; this documentation and the paper are under CC BY-NC 4.0.
- **Commercial use requires a licence from the author.**

Details and contact: `LICENSE-TIP.md`. This repository ships code only; model weights keep their own licences.
