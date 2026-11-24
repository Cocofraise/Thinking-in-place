# Licence of the Thinking in Place additions

This repository is llama.cpp plus a set of additions called Thinking in Place (TiP). The two parts have different licences.

## llama.cpp

Everything that comes from llama.cpp stays under the MIT licence, see `LICENSE`.

## Thinking in Place additions

Copyright (c) 2026 Vincent Mathieu Malet. ORCID [0009-0003-9945-7296](https://orcid.org/0009-0003-9945-7296).

The additions are:

- `src/models/tip.h`, in full;
- the code marked `TIP` in `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp`, and the matching declaration change in `src/models/models.h`;
- the lines marked `TIP` in `src/llama-moe-cache.cpp`, `src/llama-moe-cache.h`, `src/llama-graph.cpp`, `src/llama-graph.h` and `src/models/models.h`;
- `TIP.md`, `HF_CARD.md` and this file.

**Code.** The additions are licensed under the PolyForm Noncommercial License 1.0.0. The full text is at <https://polyformproject.org/licenses/noncommercial/1.0.0/> (SPDX identifier `PolyForm-Noncommercial-1.0.0`). In short: you may use, study, modify and share them for any non-commercial purpose, including research, teaching and personal use, as long as you keep this notice.

**Documentation and paper.** The documentation files above and the paper *Thinking in Place: Anchored Recurrent Depth and the Convergence of Modeling* are shared under Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0), <https://creativecommons.org/licenses/by-nc/4.0/>.

## Commercial use

Any commercial use of the additions requires a separate licence from the author. That includes selling them, offering them as part of a paid product or service, and running them in a commercial deployment. Contact the author at cocofraise974@gmail.com.

## Model weights

No model weights are distributed here. Each model keeps its own licence.
