# Thinking in Place (TiP)

**Upgrade the model you already run.** No bigger download, no retraining, no extra weights: one variable, and a frozen model thinks harder when the question is hard.

Thinking in Place is a patch on top of [llama.cpp](https://github.com/ggml-org/llama.cpp). It makes a frozen, pretrained hybrid model re-apply its own attention blocks several times per generated token, each pass anchored back to the block's input. You keep your GGUF.

- **Paper:** *Thinking in Place: Anchored Recurrent Depth and the Convergence of Modeling*, DOI [10.5281/zenodo.24507288](https://doi.org/10.5281/zenodo.24507288)
- **Windows + NVIDIA build:** [v1.0 release](https://github.com/Cocofraise/Thinking-in-place/releases/tag/v1.0)
- **Full documentation:** [TIP.md](TIP.md) (mechanism, settings, measurements, limits)

## What you get

Nine reasoning items with a checkable answer, six variants each, temperature 0, on a 9B model:

| Quantization | Single pass | `TIP_N=8` |
|---|---|---|
| Q5 | 47/54 | 50/54 |
| Q4 | 45/54 | 53/54 |
| Both | 92/108 | **103/108** |

Paired item by item, the loop wins 13 answers and loses 2. On a harder public set (MATH-500, levels 4–5, 158 questions) it goes from 78.5% to 84.8%.

It costs time, about 2× per token, and nothing else. With speculative decoding most of that overhead disappears.

No gain where the model is already right: do not expect it to help on a task your model already solves.

## Try it

**Windows + NVIDIA, no build needed.** Download the two archives from the [v1.0 release](https://github.com/Cocofraise/Thinking-in-place/releases/tag/v1.0), unzip them into the same folder, then:

```
set TIP_N=8
llama-server.exe -m model.gguf -c 8192 -ngl 99
```

**Any other platform.** Build llama.cpp as usual from this repository, then:

```bash
TIP_N=8 ./llama-server -m model.gguf -c 8192 -ngl 99
```

With `TIP_N` unset the graph is the stock llama.cpp graph and the output is identical to unpatched llama.cpp.

## Which models

Qwen hybrid attention / state-space models. Measured on Qwen 3.8 27B, Qwen 3.5 35B, Qwythos 9B, Ornith 9B and 35B, dense and mixture-of-experts. Other architectures are untested.

## Licence and contact

The Thinking in Place additions are shared for non-commercial use (PolyForm Noncommercial 1.0.0); llama.cpp itself stays under MIT. Commercial use requires a licence from the author. Details in [LICENSE-TIP.md](LICENSE-TIP.md).

Vincent Mathieu Malet · ORCID [0009-0003-9945-7296](https://orcid.org/0009-0003-9945-7296) · cocofraise974@gmail.com

The upstream llama.cpp README is kept unchanged in [README-llama.cpp.md](README-llama.cpp.md).
