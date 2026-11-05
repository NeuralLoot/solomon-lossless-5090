# Solomon: lossless speculative decoding of Qwen3.8-27B (4-bit) on one RTX 5090

## Summary

On our RTX 5090, **Qwen3.8-27B** in 4-bit (W4A16; the BF16 file is ~55.6 GB and does not fit in 32 GB) generated at **235.1 tokens/s** greedy on one request and **510.8 tokens/s** aggregate with four at once, **4 Nov 2026**, context **57,344**, DFlash **K = 7**. Same 16 short-bench prompts as 3 Nov; the 16 greedy strings match that speculative read. Acceptance was not recounted on this read; on the same short bench on 3 Nov the server kept about **5.49** tokens per step.

**Lossless** here means the speculative server’s **greedy** output matches **plain greedy** of this same 4-bit model. The drafter proposes. The target accepts or rejects. A rejected draft is never spoken. It does **not** mean 4-bit equals BF16.

## Hardware and model

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, 32 GB (serving card of our two) |
| Model | Qwen3.8-27B-OBLITERATED (architecture class `qwen3_5_text` on Hugging Face — not “Qwen3.5”) |
| Weights | W4A16 (int4, group 128, Marlin), ~16.4 GB |
| Engine | vLLM, compiled decode, CUDA graphs wide enough for eight sequences at K = 7 |
| Speculator | DFlash, **K = 7** |
| Sampling | greedy, temperature 0 |

## Latest bench (4 Nov 2026)

Date: 4 Nov 2026. Context window: **57,344**. Wall clock includes prefill. Same 16 prompts and 384 new tokens as the 3 Nov short bench. The 16 greedy strings match that 3 Nov speculative read.

| Read | What it is | Result |
| --- | --- | --- |
| Short, 1 request | 16 prompts, 384 new tokens, thinking off. | **235.1 tok/s** (4,768 tokens in 20.28 s). Per request 237.0 tok/s. |
| Short, 4 requests | Same 16 prompts, 4 at a time. | **510.8 tok/s** aggregate (4,768 tokens in 9.33 s). Per request 139.9 tok/s. The batch finishes sooner than one-at-a-time. Each request is slower than when it has the card alone. |

Steps were not recounted on this read. On the same short bench on 3 Nov: **5.49 tokens/step**, draft acceptance 0.642; per-position 0.887, 0.800, 0.701, 0.621, 0.547, 0.493, 0.443.

This read did **not** reload a plain (non-speculative) server, so it did not repeat the plain-versus-speculative identity comparison. The 16 strings match the previous speculative server. The identity result we stand on is the 16 Oct measurement below.

### Same bench, earlier the same week

| Date | 1 request | 4 requests | What changed |
| --- | ---: | ---: | --- |
| 3 Nov, graphs only wide enough for one request | 180.8 tok/s | 67 tok/s | Four-at-once fell off the compiled graph onto eager execution. |
| 3 Nov, graphs wide enough for four requests at K = 7 | **210.9 tok/s** (22.6 s) | **350.5 tok/s** (13.6 s) | Acceptance unchanged at 5.49 tokens/step. |
| 4 Nov, before the prefill index | 193.0 tok/s | — | Same kernels, a heavier prefill check. |
| 4 Nov, atomic reduction only on 16-row calls | 193.1 tok/s | **491.6 tok/s** | One request did not move. Four-wide did. |
| 4 Nov, question index and cached row center | **235.1 tok/s** | **510.8 tok/s** | Same 16 strings as 3 Nov. |

Atomic reduction on every row was measured the same evening at 181.6 and 462.2 tok/s and was not kept. Widening captured graphs from 64 to 256 tokens measured 238.5 and 515.7 tok/s, same 16 strings, and used more graph memory — treated as noise and not kept.

### Other reads, still dated

These were not re-run on 4 Nov.

| Read | Date | Result |
| --- | --- | --- |
| Reasoning, 1 request, thinking on, max 1,536 new tokens, prompts about 285 tokens | 3 Nov | **184.9 tok/s** (11,092 tokens in 60.0 s). Per request 181.8 tok/s. **5.41 tokens per step.** Per-position acceptance 0.880, 0.784, 0.687, 0.609, 0.537, 0.483, 0.431. |
| 16k context, 4 prompts, measured prompt 16.5k–18.0k tokens | 3 Nov | Prefill **75–79 s** each. Three answers ran the full 256 tokens: **32.8, 55.9, and 37.2 tok/s**. The fourth answer ended at 18 tokens, so it is not a decode-speed sample. Draft stats: **3.82 tokens/step**, acceptance 0.403. |
| Clean short bench, window 49,152, same checkpoint, same K = 7 | 23 Oct | **299.5 tok/s** (a second merged candidate the same day: 293.7). Thinking on: **258–261 tok/s**. Acceptance **5.58** tokens/step. |

The drafter’s keep rate on the short bench did not fall between 23 Oct (5.58) and 3 Nov (5.49). The wall clock did. The window grew from 49,152 to 57,344, and the prefill path grew a check that the 299 tok/s process did not have. The 4 Nov index cut that check. Decode itself was already near the weight-read floor; the 4 Nov gain is the prefill.

## Comparison with public numbers

For context, these are figures already published for Qwen3.8-27B-class models on a single RTX 5090. They use different quants, engines, and context windows, so treat them as reference points rather than a head-to-head. Their meters are often decode-only; ours include prefill in the short-bench wall clock.

**Plain quantised decode, ~115 tok/s.** Reports circulating in early Nov 2026 (e.g. a video by [@pranjalsoni_](https://x.com/pranjalsoni_/status/2089317250717348338), shared by [@Oluwaphilemon1](https://x.com/Oluwaphilemon1/status/2117805254112621022)) put Qwen3.8-27B at about **115 tok/s** generation on a 32 GB 5090. A quantised build is required there too. Our own batch-invariant plain baseline (16 Oct) was much lower, at 38 tok/s, so the gap to 235.1 comes from the drafter, not a faster plain decode.

**Lossless speculative decoding in llama.cpp.**
A llama.cpp fork (LukasParke, DFlash2, n=7, rejection sampling, IQ4_XS, f16 KV, context about 131k) is the clearest public lossless set on one RTX 5090:

| Published workload | Published tok/s | Our closest read |
| --- | ---: | --- |
| Plain decode, this model family, 5090, 32 GB | ~115 | Our batch-invariant plain baseline on 16 Oct was **38** tok/s. Their plain band on the llama.cpp fork is ~70–85. |
| Math / structured, lossless DFlash2 | 293–320 | **293.7–299.5** on 23 Oct, clean short bench, window 49,152. Current server, 4 Nov, same short bench, window 57,344: **235.1**, prefill included. |
| Code | 212–258 | **258–261** thinking-on, 23 Oct. **185** thinking-on, 3 Nov, window 57,344. |
| Open prose | 114–127 | **235** short greedy, 4 Nov, thinking off. **185** thinking-on, 3 Nov. |
| Document copy, long context | up to ~380 | Not measured as a copy task. Our 16k read on 3 Nov is prefill 75–79 s, then 33, 56, and 37 tok/s on the three answers that ran 256 tokens. |
| 4 streams, aggregate | ~285 | **510.8** aggregate and **139.9** per request, 4 Nov, prefill included. Not the same meter as a decode-only 285. On 16 Oct, 32k window, 4-wide aggregate was 440, with 16/16 identity. Different window. |
| MTP, lossless | ~130–160 | Our invariant MTP with 2 draft tokens was about **90** tok/s (16 Oct). Faster MTP numbers we measured were not batch-invariant. |

Same idea on both sides: DFlash2 proposes 7 tokens, the target verifies, rejected drafts are not spoken, so the greedy string matches normal decoding of that quant. Theirs is IQ4_XS in llama.cpp. Ours is W4A16 in vLLM, with the extra constraint of **batch invariance**: a second request must not flip a token. Their context is about 131k; ours is 57,344.

We are not claiming to be the fastest. On a clean 23 Oct read we were in their top band (about 294–300 tok/s). The server that is up now is slower on one request than that read, at the same keep rate, because the step is heavier. The 4 Nov one-request short bench is above their open-prose band. The 4 Nov four-request aggregate is above their 4-stream aggregate, and our wall clock includes prefill while theirs is a decode meter.

## Earlier measurement (16 Oct 2026)

Same kind of 5090. Window **32,768**, not 57,344. DFlash K = 7. Compiled, CUDA graphs, batch-invariant target. Greedy.

| Setup | 1 request | 4 concurrent | Accepted tokens per step | Greedy identity vs plain |
| --- | ---: | ---: | ---: | --- |
| Plain, batch-invariant | 38.0 tok/s | 132.8 tok/s | 1 | the reference |
| MTP, 2 draft tokens | ~90 tok/s | ~254 tok/s | ~1.9 | 16/16 |
| DFlash, first measure | 177 tok/s | 440 tok/s | 5.69 | 16/16 at both concurrencies |
| DFlash, after the matmul tune | **217.5 tok/s** short | KV pool was retuned | 5.35 on reasoning | 16/16 |
| DFlash, reasoning, thinking on, ~1.4k tokens | **208.7 tok/s** | — | 5.35 | same stack |
| DFlash, ~16k document question | 133.5 tok/s | — | ~3.0 | 4/4 |
| DFlash, later that evening, a bit less GPU reserved | **226.6 tok/s** short | — | 5.70 | 16/16 |

Also that day, after the last kernel fix: plain at 4 concurrent matched plain at 1 on 16/16, and DFlash at 4 matched plain at 1 on 16/16. Eight long prompts (~3k tokens, prefix-cache hits, cold and warm, 4 concurrent) matched the uncached single-request run, 8/8.

A block of 15 was measured and not kept. Short prompts hit 295 tok/s, reasoning fell to 184 tok/s, because the drafter had not been trained for positions 8–15. Production stayed at K = 7.

Use the 16 Oct table as the controlled before/after, where plain and speculative ran under the same kernels. Use 4 Nov for what the serving 5090 does on the short bench right now.

## Design

### Fit

A 27B BF16 checkpoint is about 55.6 GB. One 5090 has 32 GB. We serve W4A16, about 16.4 GB of weights, and spend the rest on the KV cache and a small drafter (about 1.2 GB). The hybrid trunk is full attention plus Gated DeltaNet. Thinking stays on for real answers.

Quantisation is a real change in weight precision. The speed method below does not undo that. It only guarantees that the fast path and the slow path of **this** checkpoint pick the same greedy token.

### Why it is faster than a plain decode

A plain step reads the weights to emit one token. On this card that read is most of the step.

1. The drafter (DFlash2, a block drafter built for Qwen3.8-27B: five layers and a candidate selector) reads the target’s hidden states and proposes up to 7 tokens. Our trunk has extra blocks, so the hidden-state taps are **[5, 20, 35, 49, 64]**, not the stock **[5, 19, 33, 47, 61]**. The stock drafter transferred with that remap and no fine-tune. A drafter we trained ourselves matched it on the live benches and was not swapped in.
2. The target scores the whole block in one forward. The weights are read once for the block, not once per token.
3. Greedy verify keeps the longest prefix that matches the target’s own argmax. The rest is dropped.

On the short bench the first positions are accepted most of the time (about 0.89, then 0.80, 0.70, …). The step returns about **5.4–5.5 tokens** instead of 1. That is the speedup. A one-layer MTP head only returned about 1.9 tokens per step, which is why the block drafter is the one we serve.

### The compiled graph

A verify of K = 7 is 8 tokens. A graph that only holds one request drops every wider batch onto eager execution. That was the 67 tok/s four-at-once read on 3 Nov. The graph now holds 64 tokens, which is eight sequences at K = 7, so the four-request read stays on the graph. The prefill chunk stays 8,192 tokens.

### The prefill check

Decode replays a CUDA graph, so a Python check wrapped around the model does not run on the compiled decode step. It does run on prefill. On 4 Nov that check was scanning a route table and re-centering a table of rows on every prompt. An index over the tokens, plus a cached center, cut that work. One request moved from 193.0 to **235.1** tok/s. Four requests moved from 491.6 to **510.8**. The 16 strings did not change.

### Why the extra tokens are the model’s own

In bf16, a token’s logits depend on the shape of the pass that computed them: how many query rows, who else is in the batch, whether a reduction was rounded between tokens, whether a matrix split across a different number of processors. A near-tie flips. Early speculative runs on this hybrid matched plain greedy on 9 of 16 prompts, and 7 of 16 with four requests in flight.

Each of the following was one measured break, fixed by one change, and re-checked:

1. **Gated DeltaNet was outside the batch-invariant kernels.** With packed recurrent decode off, a one-token step and a verify step use the same per-token kernels. Declaring that backend invariant was the first fix (11/16 identical).

2. **The recurrent state was bf16.** Plain decode rounded it every token. A verify carried it in fp32 across the drafts. The state is stored in **fp32**, so both paths round at the same time.

3. **Compiled RMSNorm was retuned per batch shape.** Both norm classes go through one batch-invariant norm kernel. The check is bitwise across batch shapes, and within one bf16 unit of the native norm on the residual form.

4. **Async scheduling made the state bookkeeping depend on timing.** The same process matched 14/16 of its own outputs on a cold start and 16/16 once warm. Async scheduling is off. Cold and warm then match.

5. **The 4-bit matrix kernel is row-invariant only up to 16 rows.** Wider calls split their reduction differently as the row count changes. Calls wider than 16 rows are executed as 16-row calls. After that, plain-at-4 matched plain-at-1, and speculative-at-4 matched plain-at-1, 16/16. A verify of K = 7 is 8 rows, so the usual one-request step does not take the split. The 16-row atomic reduction is used on those wider calls. It left the one-request read unchanged and raised the four-request read.

6. **Attention must follow the token, not the other requests.** Multi-token verify was stuck on too few processors under single-split attention (about 2.1 ms per layer at 16k context). Decode rows use a split-KV kernel. Prefill rows stay on the other kernel. A mixed batch runs each row on its own path. The tile plan is fixed from the server-wide maximum query length, not from whoever is longest in this batch. Partial sums are grouped by position, so a token at cache length *p* matches whether it is alone or first in a verify block. Check: 57/57 positions around every alignment boundary, identical at query lengths 1, 2, and 3.

7. **Gated DeltaNet prefill is a 64-token grid.** The KV block is 64 so a prefix-cache resume lands on that grid. Eight long prompts matched the uncached single-request run, including with other requests and with the cache warm.

8. **The batch-invariant matrix tile is chosen by the weight shape `(N, K)`, never by the row count `M`.** The default tile spent about 10 ms of a verify step on many small products and the vocabulary projection. A tile per weight shape cut that to about 3.5 ms and matched the default kernel bit for bit, including across batch sizes. That was the jump from about 177 tok/s to about 217 tok/s on the 32k-window day. A tile that depended on `M` would have made a 1-row decode and an 8-row verify disagree.

Greedy verify is the argmax of the target’s fp32 logits, the same reduction plain decode uses. The drafter is allowed to be wrong. It is not allowed to replace a token the target would not have chosen, and the kernel that scores the token does not change shape when the batch or the draft block changes.

### Left off on purpose

- Copying a matching span out of the prompt (“lookup drafting”) cut short-prompt speed from 174 to 92 tok/s and barely helped 16k. Off.
- K = 15 was faster on easy text and slower on reasoning, and a block-16 drafter did not fit the 57,344 window. Off.
- Draft trees need a Gated DeltaNet tree rule that is not in the vLLM version we serve. Off.
- Async scheduling stays off.
- Atomic reduction on every row slowed the one-request read (181.6 tok/s). Off. Atomic reduction on the 16-row calls stays on.
- Captured graphs through 256 tokens did not move the short bench enough to pay for the extra graph memory. Off.
- The FP4 tensor-core product is slower at decode width and does not match the served int4 values. Off.

## What we do not claim

- Not “4-bit equals the official BF16”.
- Not “299.5 tok/s” for the server that is up now. That was 23 Oct, 49,152 context.
- Not “440 tok/s” or “227 tok/s” as current. Those were 16 Oct, 32k context.
- Not a fresh plain-versus-speculative identity rerun on 4 Nov. That comparison needs a second, plain server. The identity numbers are from 16 Oct: 16/16, including with 4 requests in flight. The 4 Nov short bench matches the 3 Nov speculative strings, 16/16.
- Reasoning with thinking on is still the 3 Nov dated read: **184.9 tok/s**, 5.41 tokens/step. Not re-run on 4 Nov.
- The 16k decode samples of 33, 56, and 37 tok/s (3 Nov) are a generation speed for long context. Prefill on that read was 75–79 s, and one of the four answers ended at 18 tokens. The drafter only sees 2,048 tokens of context, so long prompts accept fewer drafts (3.82 per step on that read, against 5.49 on the short bench).
