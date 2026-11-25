# Thinking in Place (TiP) — training-free recurrent depth for llama.cpp

**More reasoning from the model you already run, with no extra weights to load.** Same GGUF, same quantization: the gain is paid in time (about 2× per token), not in a bigger model.

A patch on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) that lets a **frozen, pretrained** hybrid model re-apply its own attention blocks several times per generated token. No fine-tuning, no adapters, no extra weights: you load the same GGUF and set one environment variable.

Paper: *Thinking in Place: Anchored Recurrent Depth and the Convergence of Modeling* (Vincent Mathieu Malet, ORCID [0009-0003-9945-7296](https://orcid.org/0009-0003-9945-7296)), DOI [10.5281/zenodo.24507288](https://doi.org/10.5281/zenodo.24507288).

## What it does

Each selected full-attention block (norm → attention → residual → norm → FFN → residual) is iterated `R` times on its own hidden state. Every iteration after the first is pulled back toward the block's original input, the **anchor**:

```
s_0     = block input
in_t    = (1 - beta) * s_t + beta * s_0      (t > 0)
s_{t+1} = Block(in_t)
```

Four rules keep this safe on a model that was never trained to be iterated:

1. **Anchor every iteration.** Without the blend the iterate drifts out of the region the layer knows. The anchor strength follows the budget, `beta = 1 - φ⁻¹ / N`: the more iterations are allowed, the more gently each step moves.
2. **Spend depth where it matters.** Iterations are allocated per layer by a law of depth, `R(l) = max(1, round(N·√(l / l_max)))`. The deepest attention layer receives the full budget `N`, shallow layers stay close to one pass.
3. **Loop only where a token is predicted.** Decode steps and short verification batches (≤ 32 tokens per sequence) are looped. Prompt processing runs single-pass.
4. **Never loop a recurrent layer.** On hybrid attention / state-space models the gated-delta-net layers carry a sequential state that must advance exactly once per token. They always run once.

K/V entries of the current positions are rewritten at each iteration; the last iteration is what later tokens attend to.

## Quick start

**Windows + NVIDIA, no build needed:** download the two archives from the [v1.0 release](https://github.com/Cocofraise/Thinking-in-place/releases/tag/v1.0) (program, and the CUDA 12 runtime if you do not have the CUDA toolkit), unzip them into the same folder, then run `set TIP_N=8` followed by `llama-server.exe -m model.gguf -c 8192 -ngl 99`. The binary was checked on one machine only (RTX 4070); reports from other setups are welcome.

Build llama.cpp as usual, then:

```bash
TIP_N=8 ./llama-server -m model.gguf -c 8192 -ngl 99
```

With `TIP_N` unset (or `1`) the graph is the stock llama.cpp graph, and the output is identical to unpatched llama.cpp.

## What it buys

The loop helps where a single pass is fragile, and only there. Nine reasoning items with a checkable answer, six variants each, temperature 0, reasoning enabled, on a 9B model at its shipped quantization (Q5) and one step lower (Q4):

| Model | Single pass | `TIP_N=8` |
|---|---|---|
| Q5 | 47/54 | 50/54 |
| Q4 | 45/54 | 53/54 |
| Both | 92/108 | 103/108 |

Paired item by item, the loop wins 13 answers and loses 2 (exact two-sided p = 0.007). Seven of the nine items are at ceiling in both settings; the whole gain comes from the two items the single pass gets wrong most often (a date computation, 3/12 → 8/12, and a two-street probability, 7/12 → 11/12). The variants are perturbations of nine statements, not independent questions.

On two public benchmarks, same model and settings:

| Benchmark | Single pass | `TIP_N=8` | Paired (won / lost) |
|---|---|---|---|
| GSM8K, first 200 test questions | 195 (97.5%) | 192 (96.0%) | 2 / 5 |
| MATH-500, levels 4–5 with a plain numeric answer (158) | 124 (78.5%) | 134 (84.8%) | 19 / 9 |

GSM8K is saturated for this model: nothing to recover, and the loop moves a few answers either way. On the harder MATH-500 subset it gains 6.3 points (exact two-sided p = 0.087: a direction, not yet an established difference at this size). No gain where the model is already right, a gain where it is not. Do not expect the loop to help on a task your model already solves.

Going down the quantization ladder shows where the loop stops helping:

| Quantization | Single pass | `TIP_N=8` | |
|---|---|---|---|
| Q5 | 47/54 | 50/54 | capacity present, mostly at ceiling |
| Q4 | 45/54 | 53/54 | present and fragile: recovered |
| Q3 (re-quantized, degenerate) | 10/27 | 9/27 | lost: not recovered |
| Q2 | 0/27 | 0/27 | absent: not filled |

The loop does not bring the probabilities of a quantized model closer to the original (same distance with and without it, on the dense and on the MoE model). It changes how often the reasoning lands.

## Supported models

The Qwen3.5-style hybrid family as implemented by llama.cpp:

| File | Models |
|---|---|
| `src/models/qwen35.cpp` | dense hybrid (e.g. 9B, 27B) |
| `src/models/qwen35moe.cpp` | mixture-of-experts hybrid (e.g. 35B) |

Tested checkpoints: Qwen 3.8 27B, Qwen 3.5 35B, Qwythos 9B, Ornith 9B and 35B. The accuracy tables above are on Qwythos 9B. The loop is compatible with the Qwen hybrid architecture in general; other architectures are untested.

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
| `TIP_BETA` | an explicit anchor strength in `[0,1]`, overrides the law | derived from the budget |
| `TIP_LAYERS` | comma-separated layer indices to loop | all full-attention layers |
| `TIP_FREEZE_ROUTE` | MoE: `1` keeps the experts chosen at the first iteration, `0` routes again at each iteration | `1` |
| `TIP_FFN_LAST` | `1`: run the FFN only on the last iteration (changes the fixed point) | `0` |
| `TIP_DECODE_MAX` | largest batch (tokens per sequence) still looped | `32` |
| `TIP_PREFILL` | `1`: loop prompt-processing batches too | `0` |
| `TIP_ADAPTIVE_TAU` | per-token freeze once `cos(s_t, s_{t-1}) ≥ tau` | `0` (off) |
| `TIP_ADAPTIVE_SOFT` | width of a soft (sigmoid) freeze instead of a hard one | `0` |

Settings kept for diagnosis and experiments, not for regular use:

| Variable | Meaning | Default |
|---|---|---|
| `TIP_BETA_LAW` | `budget`: `beta = 1 - 0.618 / N`; `ref`: `beta = 1 - 0.618 / 12 = 0.9485` whatever the budget; `layer`: `beta(l) = 1 - 0.618 / R(l)`, from each layer's own iterations | `budget` |
| `TIP_BETA_SCHEDULE` | `linear`: anneal `beta` from `TIP_BETA_MIN` across iterations | off |
| `TIP_BETA_MIN` | start of the anneal | `0.3` |

With `TIP_N=8` the anchor is `1 - 0.618 / 8 = 0.923`; the accuracy tables above were run with that value passed explicitly (`TIP_N=8 TIP_BETA=0.92275`). Generation at temperature 0 is deterministic for a given value, but a change in the sixth decimal of `beta` is enough to change individual texts, so use the explicit value to reproduce them token for token. The earlier measurements of the paper use `TIP_BETA=0.9485`, the same law at the reference budget of 12. On the accuracy bench the per-layer law did not match the other two (23/27 against 25/27 and 26/27), which is why it is kept as a diagnostic.

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

Internal evaluations, small n, one architecture, quantized checkpoints, consumer hardware. Holdout replication is pending.

- The square-root schedule at budget 8 matched a uniform budget of 12 on blind panels. A uniform budget of 16 was worse than both.
- Per-layer probes show the iterate settling within about three iterations.
- Long unguarded generations collapsed into repetition far less often under the anchored loop.
- The reasoning benefit appears when the model actually deliberates. A model that leaves its reasoning block empty is neither helped nor harmed. Forcing reasoning off on a model trained to deliberate degrades the single pass, and the loop amplifies that.
- Sampling temperature is not amplified by the loop: across 3,024 positions replayed along identical prefixes the gap to the single pass is the same at T = 0, 0.7 and 1.0, and long generations show no repetition collapse, with a fixed or a per-layer anchor.

## Limits

- The loop is an **amplifier, not a corrector**: it sharpens what the underlying configuration already does, including its instabilities.
- More iterations are not monotonically better. Past the peak, quality regresses.
- Prompt processing is not looped, so tasks whose difficulty lies in reading a long input get no benefit.
- One architecture (Qwen hybrids), five checkpoints. Pure-attention stacks have no state-space majority to bound the cost.
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

Details: `LICENSE-TIP.md`. Contact: cocofraise974@gmail.com. This repository ships code only; model weights keep their own licences.

## How this was made

Research, code and writing were carried out with AI assistance. The author designed the mechanism and directed the experiments, and has read the results, the generated outputs and the conclusions, for which he takes responsibility. The prose was drafted with AI assistance and has not been reviewed word by word by the author.
