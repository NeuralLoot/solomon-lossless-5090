# Public compares — Qwen3.8-27B on one RTX 5090

Every row below is a real public report. **None of them is apples-to-apples with ours**: meters, quants, engines, contexts, and concurrency all differ. We show them so readers can see the field, not to crown a winner.

Chart: [assets/solomon-compare.png](assets/solomon-compare.png) (1 request) · [assets/solomon-load.png](assets/solomon-load.png) (under load) · script `assets/make_charts.py`.

## What "lossless" means for us

Speculative greedy output is **token-for-token identical to plain greedy of the same 4-bit checkpoint**, and a **second in-flight request cannot flip a token** (batch-invariant kernels). Last check: 16/16 prompts identical, including 4-wide. It is **not** a BF16 match. The cost: plain greedy on invariant kernels runs ~38 tok/s; DFlash2 K=7 buys the speed back without changing output.

## 1 request

| Source | Quant | Engine | Meter (decode-only?) | Batch-invariant? | tok/s | Notes / what "lossless" means there |
| --- | --- | --- | --- | --- | ---: | --- |
| **Ours (Solomon), thinking off** | W4A16 int4 g128 sym GPTQ (Marlin), bf16 act | Own serving stack, DFlash2 K=7 | **Wall clock, prefill included**, 16 prompts, temp 0, replies not cut, window 57,344 | **Yes** (checked 16/16 incl. 4-wide) | **290.5** | Identical to plain greedy of the same 4-bit checkpoint. ~5.67 tokens kept/step. |
| **Ours, thinking on** | same | same | same | **Yes** | **228.7** | Reasoning + answer in one generate. ~4.74 kept/step. |
| Ours, plain greedy (no drafter) | same | same, invariant kernels | wall clock | Yes | ~38 | The invariance tax. |
| LukasParke llama.cpp fork — math / structured | IQ4_XS GGUF, fp16 KV | llama.cpp DFlash2 n=7, rejection sampling | Often **decode-only** | Not stated | 293–320 | Context ~131k. Published as lossless DFlash2 (spec-decode in principle); no batch-invariance claim. |
| LukasParke — open prose | same | same | often decode-only | Not stated | 114–127 | Same fork. |
| LukasParke — plain decode, this family | GGUF | llama.cpp | decode-only | — | ~115 | Published plain reference. |
| [gratex HF card](https://huggingface.co/gratex/Qwen3.8-27B-DFlash2-W4A16-g128-sym-GPTQ) | **W4A16 g128 sym GPTQ target + W4A16 drafter (our quant family)** | vLLM DFlash2, spec=7 | Throughput harness, C=1, unique random seeds (no prefix-cache hits), 3 runs | Not stated | 260–270 | 260 fp8 KV / 270 bf16 KV. No-spec C=1: 88.8. No lossless/identity statement on card. |
| [darksidewalker](https://github.com/darksidewalker/qwen3.8-27b-sglang-dspark-blackwell) | NVFP4 weights, FP8 KV | SGLang DFlash2 (block 8) | **Live-traffic median** from Grafana (peak 277) | Not stated | 221 | Context 80k configured. No lossless statement found. |
| [Gabe Ortiz / localLLMrig](https://localllmrig.com/interviews/gabe-ortiz-rtx-5090-q27) | 4-bit default tier (17.00 GB) | q27 (Quasar) + DFlash2 | **Self-reported**, Claude Code traffic, ~4.1 accepted/round | Not stated | ~220 | Plain ceiling ~105. No lossless statement. |
| [GenerationAI (Medium)](https://medium.com/@GenerationAI/one-draft-model-two-verdicts-dflash-2-takes-qwen3-8-27b-f16ab29004d0) | Not stated in the excerpt we could read | llama.cpp DFlash2 | Decode benchmark | Not stated | 215 | Up from ~146 with MTP. Article calls DFlash2 "lossless under greedy decoding" (the algorithm), not a measured identity check. |
| [Suhail Mohebi](https://suhailmohebi.com/insights/rtx-5090-llm-serving/) — SGLang DFlash2 | NVFP4 (ModelOpt), FP8 KV | SGLang | Sustained output, matched 32k suite, C=1, warm server | Not stated | 139.7 | Profiles must pass recall/tool/vision/stability gates; FP8 KV has no matched BF16 comparison (author's note). |
| Suhail — SGLang DSpark | same | SGLang | same | Not stated | 110.2 | |
| Suhail — vLLM native MTP | same | vLLM | same | Not stated | 100.8 | A later isolated MTP A/B reported "unchanged outputs". |
| Suhail — vLLM baseline | same | vLLM | same | — | 84.5 | No speculation; his fidelity reference. |

## Under load (aggregate)

| Source | Quant | Engine | Meter | Concurrency | Batch-invariant? | tok/s |
| --- | --- | --- | --- | ---: | --- | ---: |
| **Ours, thinking off** | W4A16 | own stack, DFlash2 K=7 | wall, prefill in, not cut | 4 | **Yes** | **614.3** |
| **Ours, thinking on** | W4A16 | same | same | 4 | **Yes** | **467.4** |
| LukasParke | IQ4_XS | llama.cpp DFlash2 | decode meter | 4 streams | Not stated | ~285 |
| gratex | W4A16 | vLLM DFlash2 spec=7 | harness | **8** | Not stated | 776–868 (bf16 / fp8 KV) |
| gratex | W4A16 | vLLM DFlash2 spec=3 | harness | **8** | Not stated | 995–1,013 (fp8 / bf16 KV) |

gratex runs twice our concurrency and states no token-identity check; more concurrency buys raw throughput. We trade some of it for the guarantee that request #2 can't change request #1's tokens.

## How we differ (short)

- **Meter:** ours is wall clock with prefill, replies never cut. Several public numbers are decode-only, live-traffic medians, or self-reported.
- **Lossless is measured, not assumed:** spec output checked identical to plain greedy (same 4-bit checkpoint), 16/16 incl. 4-wide. Others rely on the algorithm's greedy guarantee or pass/fail quality gates.
- **Batch-invariant:** no public row states it. Without it, a second request can shift float reduction order and flip a token.
- **We pay a tax for it:** plain invariant greedy ~38 tok/s vs ~85–115 plain elsewhere. The drafter earns it back.
- **Quant/context:** W4A16 GPTQ at a 57,344 window vs IQ4_XS ~131k, NVFP4 32k–80k, etc.
