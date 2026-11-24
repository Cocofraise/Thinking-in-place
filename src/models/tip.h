#pragma once

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Thinking in Place - Copyright (c) 2026 Vincent Mathieu Malet. Non-commercial use only; commercial use
// requires a licence from the author. See LICENSE-TIP.md.

// TIP (Thinking in Place) — anchored recurrent depth, training-free.
//
// Each selected FULL-ATTENTION layer block (norm -> attn -> residual -> norm -> ffn -> residual)
// is iterated R times on its own hidden state at inference, on frozen weights:
//
//     s_0     = layer input
//     in_t    = (1 - beta) * s_t + beta * s_0        (t > 0 : input injection, the anchor)
//     s_{t+1} = Block(in_t)
//
// Recurrent-state layers (gated delta net / linear attention) are never looped: their sequential
// state must advance exactly once per token.
//
// K/V of the current positions are rewritten at every iteration; the last (converged) iteration
// is what stays in the cache for future tokens.
//
// All knobs are environment variables, read once. TIP_N=1 (default) builds the stock graph.
//
//   TIP_N             R, iterations of the deepest looped layer            (default 1 = off)
//   TIP_BETA          anchor strength in [0,1]                             (default 0.5)
//   TIP_BETA_LAW      "phi" : per-layer beta = 1 - 0.618 / R(il)           (default off)
//   TIP_BETA_SCHEDULE "linear" : anneal beta from BETA_MIN (t=0) to beta   (default off)
//   TIP_BETA_MIN      start of the anneal                                  (default 0.3)
//   TIP_DEPTH_LAW     "sqrt" | "log" | "square" : taper R with layer index (default flat)
//   TIP_LAYERS        csv of layer indices to loop                         (default all full-attention)
//   TIP_FFN_LAST      1 : run the FFN only on the last iteration           (default 0)
//   TIP_FREEZE_ROUTE MoE only: keep the experts chosen at iteration 0     (default 1 ; 0 = re-route every iteration)
//   TIP_DECODE_MAX    largest ubatch (tokens per sequence) still looped    (default 32)
//   TIP_PREFILL       1 : loop prompt-processing batches too               (default 0)
//   TIP_ADAPTIVE_TAU  per-token freeze once cos(s_t, s_{t-1}) >= tau       (default 0 = off)
//   TIP_ADAPTIVE_SOFT width of a soft (sigmoid) freeze instead of a step   (default 0 = step)

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

struct tip_cfg {
    int   n            = 1;
    float beta         = 0.5f;
    bool  beta_phi     = false;
    bool  beta_anneal  = false;
    float beta_min     = 0.3f;
    int   depth_law    = 0;      // 0 flat, 1 sqrt, 2 log, 3 square
    bool  ffn_last     = false;
    bool  freeze_route = true;
    int   decode_max   = 32;
    bool  prefill      = false;
    float adaptive_tau  = 0.0f;
    float adaptive_soft = 0.0f;
    std::vector<int> layers;     // empty = every full-attention layer

    static float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

    static const tip_cfg & get() {
        static const tip_cfg cfg = [] {
            tip_cfg c;
            if (const char * e = getenv("TIP_N"))            { c.n = std::max(1, atoi(e)); }
            if (const char * e = getenv("TIP_BETA"))         { c.beta = clamp01((float) atof(e)); }
            if (const char * e = getenv("TIP_BETA_LAW"))     { c.beta_phi = !strcmp(e, "phi"); }
            if (const char * e = getenv("TIP_BETA_SCHEDULE")){ c.beta_anneal = !strcmp(e, "linear"); }
            if (const char * e = getenv("TIP_BETA_MIN"))     { c.beta_min = clamp01((float) atof(e)); }
            if (const char * e = getenv("TIP_DEPTH_LAW")) {
                c.depth_law = !strcmp(e, "sqrt") ? 1 : !strcmp(e, "log") ? 2 : !strcmp(e, "square") ? 3 : 0;
            }
            if (const char * e = getenv("TIP_FFN_LAST"))     { c.ffn_last = atoi(e) != 0; }
            if (const char * e = getenv("TIP_FREEZE_ROUTE")) { c.freeze_route = atoi(e) != 0; }
            if (const char * e = getenv("TIP_DECODE_MAX"))   { c.decode_max = atoi(e); }
            if (const char * e = getenv("TIP_PREFILL"))      { c.prefill = atoi(e) != 0; }
            if (const char * e = getenv("TIP_ADAPTIVE_TAU")) { c.adaptive_tau = (float) atof(e); }
            if (const char * e = getenv("TIP_ADAPTIVE_SOFT")){ c.adaptive_soft = (float) atof(e); }
            if (const char * e = getenv("TIP_LAYERS")) {
                for (const char * p = e; *p; ) {
                    c.layers.push_back(atoi(p));
                    while (*p && *p != ',') { p++; }
                    if (*p == ',') { p++; }
                }
            }
            return c;
        }();
        return cfg;
    }

    // Iterations for layer `il`. `il_max` = index of the deepest full-attention layer.
    // `n_seq_tokens` = tokens per sequence in this ubatch: large batches are prompt processing,
    // nothing is sampled from them except their last position, so they run once unless forced.
    int iterations(int il, int il_max, bool is_full_attn, int64_t n_seq_tokens) const {
        if (n <= 1 || !is_full_attn) {
            return 1;
        }
        if (!layers.empty() && std::find(layers.begin(), layers.end(), il) == layers.end()) {
            return 1;
        }
        if (!prefill && n_seq_tokens > decode_max) {
            return 1;
        }
        if (depth_law == 0 || il_max <= 0) {
            return n;
        }
        float raw, raw_max;
        switch (depth_law) {
            case 1:  raw = sqrtf((float) il);        raw_max = sqrtf((float) il_max);          break;
            case 3:  raw = (float) il * (float) il;  raw_max = (float) il_max * (float) il_max; break;
            default: raw = logf((float) std::max(il, 1)); raw_max = logf((float) il_max);       break;
        }
        const float scale = raw_max > 0.0f ? (float) n / raw_max : 1.0f;
        return std::max(1, (int) std::lround(raw * scale));
    }

    // Anchor strength at iteration t of a layer that runs R iterations.
    float beta_at(int t, int R) const {
        const float layer_beta = (beta_phi && R > 1) ? 1.0f - 0.618f / (float) R : beta;
        if (beta_anneal && R > 1) {
            return beta_min + ((float) t / (float) (R - 1)) * (layer_beta - beta_min);
        }
        return layer_beta;
    }
};
