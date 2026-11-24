# Thinking in Place (TiP) — training-free recurrent depth for llama.cpp

A patch on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) that lets a **frozen, pretrained** hybrid model re-apply its own attention blocks several times per generated token. No fine-tuning, no adapters, no extra weights: you load the same GGUF and set one environment variable.

Paper: *Thinking in Place: Anchored Recurrent Depth and the Convergence of Modeling* (Vincent Mathieu Malet, ORCID [0009-0003-9945-7296](https://orcid.org/0009-0003-9945-7296)).

## What it does

Each selected full-attention block (norm → attention → residual → norm → FFN → residual) is iterated `R` times on its own hidden state. Every iteration after the first is pulled back toward the block's original input, the **anchor**:

```
s_0     = block input
in_t    = (1 - beta) * s_t + beta * s_0      (t > 0)
s_{t+1} = Block(in_t)
```

Four rules keep this safe on a model that was never trained to be iterated:

1. **Anchor every iteration.** Without the blend the iterate drifts out of the region the layer knows. The anchor strength follows each layer's own budget, `beta(l) = 1 - φ⁻¹ / R(l)`: the more a layer iterates, the more gently each step moves.
2. **Spend depth where it matters.** Iterations are allocated per layer by a law of depth, `R(l) = max(1, round(N·√(l / l_max)))`. The deepest attention layer receives the full budget `N`, shallow layers stay close to one pass.
3. **Loop only where a token is predicted.** Decode steps and short verification batches (≤ 32 tokens per sequence) are looped. Prompt processing runs single-pass.
4. **Never loop a recurrent layer.** On hybrid attention / state-space models the gated-delta-net layers carry a sequential state that must advance exactly once per token. They always run once.

K/V entries of the current positions are rewritten at each iteration; the last iteration is what later tokens attend to.

## Quick start

Build llama.cpp as usual, then:

```bash
TIP_N=8 ./llama-server -m model.gguf -c 8192 -ngl 99
```

With `TIP_N` unset (or `1`) the graph is the stock llama.cpp graph, and the output is identical to unpatched llama.cpp.

## Supported models

The Qwen3.5-style hybrid family as implemented by llama.cpp:

| File | Models |
|---|---|
| `src/models/qwen35.cpp` | dense hybrid (e.g. 9B, 27B) |
| `src/models/qwen35moe.cpp` | mixture-of-experts hybrid (e.g. 35B) |

The logic and every setting live in `src/models/tip.h`. The patch also adds about fifty lines marked `TIP` to shared files of llama.cpp (`src/llama-moe-cache.cpp`, `src/llama-moe-cache.h`, `src/llama-graph.cpp`, `src/llama-graph.h`, `src/models/models.h`) so that a layer can present several expert selections in one pass and so that the graph is sized for the loop.

## What works

Checked against unpatched llama.cpp `af5c583` (2026-11-23), CUDA build, on a 9B dense model and a 35B MoE model.

| Area | Status |
|---|---|
| Loop off | same text and same first-token log-probabilities as unpatched llama.cpp, dense and MoE |
| Loop on | dense and MoE; GPU, CPU-only, and MoE experts kept in host memory (`--n-cpu-moe`) |
| Speculative decoding | `draft-mtp` and `draft-dflash` return the same text as the loop alone, 2.3 to 2.6 times faster; `draft-dflash2` runs |
| Parallel slots | 4 and 8 slots, dense and MoE, frozen and free routing |
| MoE expert cache (`--moe-cache-mib`) | frozen and free routing, with and without CPU share |
| Every setting below | exercised on both models |

## Settings

| Variable | Meaning | Default |
|---|---|---|
| `TIP_N` | budget: iterations of the deepest looped layer | `1` (off) |
| `TIP_DEPTH_LAW` | `sqrt`, `log`, `square` or `flat`: how the budget is spread over layers | `sqrt` |
| `TIP_BETA_LAW` | `phi`: `beta(l) = 1 - 0.618 / R(l)`, per layer; `fixed`: one value for all layers | `phi` |
| `TIP_BETA` | a fixed anchor strength in `[0,1]`; setting it selects `fixed` | `0.9485` when fixed |
| `TIP_LAYERS` | comma-separated layer indices to loop | all full-attention layers |
| `TIP_FREEZE_ROUTE` | MoE: `1` keeps the experts chosen at the first iteration, `0` routes again at each iteration | `1` |
| `TIP_FFN_LAST` | `1`: run the FFN only on the last iteration (changes the fixed point) | `0` |
| `TIP_DECODE_MAX` | largest batch (tokens per sequence) still looped | `32` |
| `TIP_PREFILL` | `1`: loop prompt-processing batches too | `0` |
| `TIP_ADAPTIVE_TAU` | per-token freeze once `cos(s_t, s_{t-1}) ≥ tau` | `0` (off) |
| `TIP_ADAPTIVE_SOFT` | width of a soft (sigmoid) freeze instead of a hard one | `0` |
| `TIP_BETA_SCHEDULE` | `linear`: anneal `beta` from `TIP_BETA_MIN` across iterations | off |
| `TIP_BETA_MIN` | start of the anneal | `0.3` |

The measurements of the paper use `TIP_N=8`, the `sqrt` law and a fixed `TIP_BETA=0.9485`, which is the same law evaluated at the reference budget of 12.

A weak fixed anchor is the one setting to avoid: `TIP_BETA=0.5` with a flat budget produces degenerate text.

## Expert routing inside the loop (MoE)

The expert router sits inside the looped block. This gives two distinct modes, chosen for what they do to the output.

- **Frozen (default).** The experts are chosen once, at the first iteration, and every later iteration of that token uses the same set. Only the discrete choice is frozen; the mixing weights are recomputed each iteration. It is one reader going over the same page.
- **Free (`TIP_FREEZE_ROUTE=0`).** The router picks again at every iteration. The reader changes each time.

What the two modes do, read on full outputs of the 35B with reasoning enabled:

| | Frozen | Free |
|---|---|---|
| Four Bayesian traps | correct final answer on 4 of 4; right the first time on the base-rate item | correct on 3 of 4; on the base-rate item it slips, notices, fixes, and runs out of budget |
| A trade-off to argue | conditional verdict, lists when it would switch | one committed line |
| A short tale | tighter, one idea carried to its end | longer, more images, one continuity slip left in |
| A thought experiment | one organising idea | more alternatives laid side by side |
| Hardest open prompt | deliberates without writing an answer | writes the piece |

Frozen for checking a computation, free for open-ended generation. One run per mode and per prompt: a reading, not a rate.

Speed is the same in both modes (within about 1%), except with the MoE expert cache, where frozen routing is about 8% faster because a token's experts are uploaded once instead of once per iteration.

## Cost

The loop adds compute only on full-attention layers and only on predicting passes. Measured on this port at about 13,500 tokens of context, without speculative decoding:

| Model | Single pass | `TIP_N=8`, sqrt | Uniform 8 | Uniform 16 |
|---|---|---|---|---|
| 9B | 16.2 ms/token | 34.4 (×2.1) | 43.8 (×2.7) | 75.4 (×4.7) |
| 35B MoE, experts of 16 layers on CPU | 18.5 ms/token | 31.4 (×1.7) | — | — |

On the 9B the time per token is linear in the number of block evaluations, 0.49 ms each. The loop therefore executes exactly the iterations the schedule allocates, and the square-root schedule costs about half of a uniform budget of 12 (45 block evaluations per token against 96).

With speculative decoding most of this overhead disappears: the 9B at `TIP_N=8` runs at 81 tokens/s with `draft-mtp`, faster than the single pass without speculation (59 tokens/s).

## Results reported in the paper

Internal evaluations, small n, one model family, quantized checkpoints, consumer hardware. Holdout replication is pending.

- The square-root schedule at budget 8 matched a uniform budget of 12 on blind panels. A uniform budget of 16 was worse than both.
- Per-layer probes show the iterate settling within about three iterations.
- Long unguarded generations collapsed into repetition far less often under the anchored loop.
- The reasoning benefit appears when the model actually deliberates. A model that leaves its reasoning block empty is neither helped nor harmed. Forcing reasoning off on a model trained to deliberate degrades the single pass, and the loop amplifies that.
- Sampling temperature is not amplified by the loop: across 3,024 positions replayed along identical prefixes the gap to the single pass is the same at T = 0, 0.7 and 1.0, and long generations show no repetition collapse, with a fixed or a per-layer anchor.

## Limits

- The loop is an **amplifier, not a corrector**: it sharpens what the underlying configuration already does, including its instabilities.
- More iterations are not monotonically better. Past the peak, quality regresses.
- Prompt processing is not looped, so tasks whose difficulty lies in reading a long input get no benefit.
- One model family. Pure-attention stacks have no state-space majority to bound the cost.
- With several slots, outputs are not token-identical to a single slot. This is a property of batched inference in llama.cpp and holds with the loop off as well.
- Not included: the convergence probe, per-request control of the budget, and the CUDA expert-gather kernel described in the paper.

## Related work

Looped and recurrent-depth transformers are an active line of work, mostly **trained** with the loop: the model is pretrained or fine-tuned so that a shared block can be applied several times. SenseNova-Loop-9B (November 2026) is a recent example on a model of similar shape: a block of 24 layers is passed twice, with learned two-stream residual mixing.

This patch takes the other road. Nothing is trained. It iterates individual attention blocks of an existing checkpoint, with an anchor, a depth schedule, and a rule for which passes are looped. The two approaches are not exclusive.

Direction sparked by a survey of loop-transformer papers (arXiv 2604.07822, 2604.11791, 2604.12946, 2507.10524). The anchoring, the schedule and the port are the author's own.

## Licence

Shared for non-commercial use.

- llama.cpp itself stays MIT (`LICENSE`).
- The Thinking in Place additions are under the PolyForm Noncommercial License 1.0.0; this documentation and the paper are under CC BY-NC 4.0.
- **Commercial use requires a licence from the author.**

Details and contact: `LICENSE-TIP.md`. This repository ships code only; model weights keep their own licences.
