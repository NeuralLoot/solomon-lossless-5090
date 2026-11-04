# Solomon: lossless speculative decoding of Qwen3.8-27B (4-bit) on one RTX 5090

## Summary

On our RTX 5090, **Qwen3.8-27B** in 4-bit (W4A16; the BF16 file is ~55.6 GB and does not fit in 32 GB) generated at **183 tokens/s** with thinking on and **181 tokens/s** greedy with thinking off, 3 Nov 2026, context **57,344**. A block drafter (DFlash, K = 7) proposes 7 tokens; the model verifies them and keeps about **5.4** per step.

**Lossless** here means the speculative server’s **greedy** output matches **plain greedy** of this same 4-bit model. The drafter proposes. The target accepts or rejects. A rejected draft is never spoken. It does **not** mean 4-bit equals BF16.

## Hardware and model

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, 32 GB (serving card of our two) |
| Model | Qwen3.8-27B-OBLITERATED (architecture class `qwen3_5_text` on Hugging Face — not “Qwen3.5”) |
| Weights | W4A16 (int4, group 128, Marlin), ~16.4 GB |
| Engine | vLLM, compiled decode, CUDA graphs |
| Speculator | DFlash, **K = 7** |
| Sampling | greedy, temperature 0 |

## Latest bench (3 Nov 2026)

Date: 3 Nov 2026, 19:47–20:00 America/Denver. Context window: **57,344**. No other request was decoding; the draft counter was still before the read started.

| Read | What it is | Result |
| --- | --- | --- |
| Short, 1 request | 16 prompts, 384 new tokens, thinking off. Wall clock includes prefill. | **180.8 tok/s** (4,768 tokens in 26.4 s). Per request 176.2 tok/s. |
| Short, 4 requests | Same 16 prompts, 4 at a time. Wall clock includes prefill. | **67.4 tok/s** aggregate (4,768 tokens in 70.8 s). Per request 17.9 tok/s. Slower than one-at-a-time on this 57k window. Not a multi-user speedup. |
| Reasoning, 1 request | 8 frontier asks, thinking on, max 1,536 new tokens, prompts about 285 tokens. | **183.1 tok/s** (11,092 tokens in 60.6 s). Per request 179.2 tok/s. **5.41 tokens per step.** Per-position acceptance 0.880, 0.784, 0.687, 0.609, 0.537, 0.483, 0.431. |
| Short, both concurrencies, draft stats | Steps counted on the server across the short read only. | 1,746 steps, **5.49 tokens/step**, draft acceptance 0.642. Per-position 0.887, 0.800, 0.701, 0.621, 0.547, 0.493, 0.443. |
| 16k context | 4 prompts, measured prompt 16.5k–18.0k tokens, 256 new tokens. Prefill separate from decode. | Prefill **71–80 s** each. Usable decode rows: **83 tok/s** and **32 tok/s**. Two later rows finished in the same instant as the warm pass, so their tok/s is not usable. Draft stats: **3.82 tokens/step**, acceptance 0.403. |

Reading the table:

- One reasoning turn with thinking on: **183 tok/s** at 5.41 accepted tokens per step.
- One short greedy generation: **181 tok/s**.
- Four at once on these short prompts was slower, not faster. A 57,344-token window spends the card on KV; four prefills at once did not pay for themselves on this read.
- The 16k read is not a clean speed number. Quote prefill (71–80 s) and the two real decode rows (83 and 32 tok/s), plus 3.82 tokens/step. Do not quote a 70.7 tok/s aggregate.

This rebench did **not** reload a plain (non-speculative) server, so it did not repeat the 16-prompt identity comparison tonight. The identity result we stand on is the 16 Oct measurement below.

### Why the latest read is slower than our 23 Oct read

On 23 Oct the same 5090, the same merged checkpoint, and the same DFlash K=7, a clean single-stream read of this same short bench was **299.5 tok/s** (merged candidate 293.7), thinking **258–261 tok/s**. Acceptance was **5.58** tokens/step. Tonight’s short read is **180.8** at **5.49** tokens/step.

The drafter still keeps about five and a half tokens per step. What changed is the step time: about **19 ms** then (299.5 / 5.58) versus about **30 ms** now (180.8 / 5.49). A dirty read that same morning, with other work on the card, was already down at 241 tok/s. The GPU was not slower. The step was busier.

Since that clean read, three things on the serving 5090 changed:

1. The context window grew from **49,152** to **57,344**. More of the 32 GB is reserved for KV. At the rebench the card was at **31.3 / 32.6 GB**, with the desktop on the same GPU.
2. Every model forward now also does an extra check that was not in the 299 tok/s process. It does not change how many tokens the drafter keeps.
3. Four-at-once tonight (**67 tok/s** wall, prefill included) is not a decode-only aggregate. The 57k pool was already full, so four prefills serialized.

## Comparison with public numbers

For context, these are figures already published for Qwen3.8-27B-class models on a single RTX 5090. They use different quants, engines, and context windows, so treat them as reference points rather than a head-to-head.

**Plain quantised decode, ~115 tok/s.** Reports circulating in early Nov 2026 (e.g. a video by [@pranjalsoni_](https://x.com/pranjalsoni_/status/2089317250717348338), shared by [@Oluwaphilemon1](https://x.com/Oluwaphilemon1/status/2117805254112621022)) put Qwen3.8-27B at about **115 tok/s** generation on a 32 GB 5090. A quantised build is required there too. Our 183 tok/s thinking-on read is about 1.6× that figure; our own batch-invariant plain baseline (16 Oct) was much lower, at 38 tok/s, so the gap comes from the drafter, not a faster plain decode.

**Lossless speculative decoding in llama.cpp.** 
A llama.cpp fork (LukasParke, DFlash2, n=7, rejection sampling, IQ4_XS, f16 KV, context about 131k) is the clearest public lossless set on one RTX 5090:

| Their workload | Their tok/s | Our closest read |
| --- | ---: | --- |
| Math / structured | 293–320 | **293.7–299.5** on 23 Oct, clean short bench, window 49,152, same 5090. Same band. |
| Code | 212–258 | **258–261** thinking-on, 23 Oct. **183** tonight, thinking on, window 57,344. |
| Open prose | 114–127 | **181–183** tonight. Ahead of that band even on the slower server. |
| Document copy, long context | up to ~380 | Not measured. We tried prompt-lookup drafting and turned it off: short prompts fell from 174 to 92 tok/s. |
| 4 streams, aggregate | ~285 | Not a number we can match tonight. Our 4-wide short read was 67 tok/s and includes prefill. On 16 Oct, 32k window, 4-wide aggregate was 440, with 16/16 identity. Different window. |
| Plain decode | ~70–85 | Our batch-invariant plain baseline on 16 Oct was **38** tok/s. The invariance work costs speed before the drafter is added. |
| MTP, lossless | ~130–160 | Our invariant MTP-2 was about **90** tok/s (16 Oct). Faster MTP numbers we measured were not batch-invariant. |

Same idea on both sides: DFlash2 proposes 7 tokens, the target verifies, rejected drafts are not spoken, so the greedy string matches normal decoding of that quant. Theirs is IQ4_XS in llama.cpp. Ours is W4A16 in vLLM, with the extra constraint of **batch invariance**: a second request must not flip a token. That is why 16 Oct could say 4-concurrent matched 1-concurrent, 16/16 — and why it is a tax their fork does not pay. Their context is about 131k; ours is 57k.

We are not claiming to be the fastest. On a clean 23 Oct read we were in their top band (about 294–300 tok/s); the server that is up tonight is slower because the step is heavier, not because acceptance fell; open-prose on tonight’s server is still above their 114–127 band.

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

Use the 16 Oct table as the controlled before/after, where plain and speculative ran under the same kernels. Use 3 Nov for what the serving 5090 does right now.

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

### Why the extra tokens are the model’s own

In bf16, a token’s logits depend on the shape of the pass that computed them: how many query rows, who else is in the batch, whether a reduction was rounded between tokens, whether a matrix split across a different number of processors. A near-tie flips. Early speculative runs on this hybrid matched plain greedy on 9 of 16 prompts, and 7 of 16 with four requests in flight.

Each of the following was one measured break, fixed by one change, and re-checked:

1. **Gated DeltaNet was outside the batch-invariant kernels.** With packed recurrent decode off, a one-token step and a verify step use the same per-token kernels. Declaring that backend invariant was the first fix (11/16 identical).

2. **The recurrent state was bf16.** Plain decode rounded it every token. A verify carried it in fp32 across the drafts. The state is stored in **fp32**, so both paths round at the same time.

3. **Compiled RMSNorm was retuned per batch shape.** Both norm classes go through one batch-invariant norm kernel. The check is bitwise across batch shapes, and within one bf16 unit of the native norm on the residual form.

4. **Async scheduling made the state bookkeeping depend on timing.** The same process matched 14/16 of its own outputs on a cold start and 16/16 once warm. Async scheduling is off. Cold and warm then match.

5. **The 4-bit matrix kernel is row-invariant only up to 16 rows.** Wider calls split their reduction differently as the row count changes. Calls wider than 16 rows are executed as 16-row calls. After that, plain-at-4 matched plain-at-1, and speculative-at-4 matched plain-at-1, 16/16. A verify of K = 7 is 8 rows, so the usual step does not take the split.

6. **Attention must follow the token, not the other requests.** Multi-token verify was stuck on too few processors under single-split attention (about 2.1 ms per layer at 16k context). Decode rows use a split-KV kernel. Prefill rows stay on the other kernel. A mixed batch runs each row on its own path. The tile plan is fixed from the server-wide maximum query length, not from whoever is longest in this batch. Partial sums are grouped by position, so a token at cache length *p* matches whether it is alone or first in a verify block. Check: 57/57 positions around every alignment boundary, identical at query lengths 1, 2, and 3.

7. **Gated DeltaNet prefill is a 64-token grid.** The KV block is 64 so a prefix-cache resume lands on that grid. Eight long prompts matched the uncached single-request run, including with other requests and with the cache warm.

8. **The batch-invariant matrix tile is chosen by the weight shape `(N, K)`, never by the row count `M`.** The default tile spent about 10 ms of a verify step on many small products and the vocabulary projection. A tile per weight shape cut that to about 3.5 ms and matched the default kernel bit for bit, including across batch sizes. That was the jump from about 177 tok/s to about 217 tok/s on the 32k-window day. A tile that depended on `M` would have made a 1-row decode and an 8-row verify disagree.

Greedy verify is the argmax of the target’s fp32 logits, the same reduction plain decode uses. The drafter is allowed to be wrong. It is not allowed to replace a token the target would not have chosen, and the kernel that scores the token does not change shape when the batch or the draft block changes.

### Left off on purpose

- Copying a matching span out of the prompt (“lookup drafting”) cut short-prompt speed from 174 to 92 tok/s and barely helped 16k. Off.
- K = 15 was faster on easy text and slower on reasoning. Off.
- Draft trees need a Gated DeltaNet tree rule that is not in the vLLM version we serve. Off.
- Async scheduling stays off.

## What we do not claim

- Not “4-bit equals the official BF16”.
- Not “440 tok/s” or “227 tok/s” for the server that is up tonight. Those were 16 Oct, 32k context.
- Not a fresh 16-prompt identity rerun tonight. That comparison needs a second, plain server. The identity numbers are from 16 Oct: 16/16, including with 4 requests in flight.
- Tonight’s 16k read: prefill 71–80 s, usable decode 83 and 32 tok/s, 3.82 tokens kept per step. The 16 Oct 16k read was 134 tok/s on a 32k window. The drafter only sees 2,048 tokens of context, so long prompts accept fewer drafts.
