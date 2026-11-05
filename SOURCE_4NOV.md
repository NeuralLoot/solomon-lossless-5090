# Qwen3.8-27B on one RTX 5090: how the speed is produced

One NVIDIA GeForce RTX 5090, 32 GB, SM 12.0. The numbers below are greedy generation of **Qwen3.8-27B-OBLITERATED** (an abliterated Qwen 3.8 27B). The checkpoint’s `config.json` says `model_type: qwen3_5_text`. That is the Hugging Face class. It is not the model version.

“Lossless” means the speculative server’s greedy string matches plain greedy of **this same 4-bit checkpoint**. The drafter proposes. The target argmax verifies. A rejected draft is never spoken. 4-bit is not BF16. The official BF16 file is about 55.6 GB and does not fit in 32 GB.

The current short bench is **4 Nov 2026**: context **57,344**, DFlash **K = 7**. Older rows are labelled with their date and their window.

## What to copy

### Weights

Serve W4A16: int4, group size 128, symmetric GPTQ, Marlin. About 16.4 GB. Activations stay bf16. The hybrid trunk is full attention plus Gated DeltaNet.

| | |
| --- | --- |
| Layers | 68 (20 full attention, 48 Gated DeltaNet) |
| Hidden / intermediate | 5,120 / 17,408 |
| Full attention | 24 query heads, 4 KV heads, head dim 256, output gate on |
| Gated DeltaNet | 16 key heads × 128, 48 value heads × 128, conv width 4 |
| Vocab | 248,320 |
| Native context | 262,144. The served window is a KV-pool choice, not the architecture. |

Int4 GEMM shapes in one layer, K then N:

| Linear | K | N | Int4 weight |
| --- | ---: | ---: | ---: |
| Q | 5,120 | 6,144 | 15.7 MB |
| K, and V | 5,120 | 1,024 | 2.6 MB each |
| Attention out | 6,144 | 5,120 | 15.7 MB |
| DeltaNet in-proj (q, k, v, z) | 5,120 | 16,384 | 41.9 MB |
| DeltaNet out | 6,144 | 5,120 | 15.7 MB |
| Gate+up | 5,120 | 34,816 | 89.1 MB |
| Down | 17,408 | 5,120 | 44.6 MB |

All 68 layers are 12.59 GB of int4. A 4-bit product of M rows does `4·M` operations per byte of weight. A K = 7 verify is M = 8, so 32 operations per byte. This card’s HBM is 1.79 TB/s (512-bit at 14,001 MHz). Decode is waiting on the weight read. A new multiply does not avoid that read.

### Drafter

DFlash2, block size 8, served at **K = 7** (the target scores 1 + 7 = 8 tokens in one forward). Five sliding-attention layers, same width as the trunk (hidden 5,120, intermediate 17,408), 32 heads, 8 KV heads, head dim 128, sliding window 32,768, about 1.2 GB in the same W4A16 layout. It reads the target hidden states at layers **[5, 20, 35, 49, 64]**. A stock 27B drafter is tapped at [5, 19, 33, 47, 61]. This trunk has extra blocks, so the taps move. The selector rank is 256, top-16.

The weights are read once for the block. Greedy verify keeps the longest prefix that matches the target argmax. On the short bench that is about **5.5 tokens per step** instead of 1. A one-layer MTP head kept about 1.9 and was slower. K = 15 was faster on easy text (295 tok/s) and slower on reasoning (184 tok/s), and a block-16 drafter did not leave room for a 57,344 window. Stay at K = 7.

Sampled requests (temperature > 0) must use the draft distribution in the rejection test (`draft_sample_method=probabilistic`). Pinning the draft probability to 1 lowers acceptance. Temperature 0 is unchanged either way: greedy verify is the target argmax, and the draft-logit scan is skipped when temperature is 0.

### Server settings

These are the settings behind the 4 Nov numbers. Leave a setting out and the identity or the speed changes, as the table of rejected runs shows.

| Setting | Value | Why |
| --- | --- | --- |
| `max_model_len` | 57344 | KV pool on one 32 GB card. fp8 KV was tried and diverged from fp16 at a long prompt. KV stays fp16. |
| `max_num_seqs` | 8 | Four concurrent verifies at K = 7, with room for one more. A large LoRA slot is about 1.7 GB and is off. |
| `max_num_batched_tokens` | 8192 | Chunked prefill. This is the profile-run peak. Raising it blows the 57,344 window. |
| `max_cudagraph_capture_size` | 64 | 8 sequences × 8 tokens. Capture 8 graphs one request and drops four-at-once onto eager (67 tok/s). Capture 256 did not pay for itself. |
| `gpu_memory_utilization` | 0.95, then clamped to whatever still fits | A cold compile can miss the window by tens of MB. A second boot from the cache fit 57,344 with about 7.5 GiB of KV. If the boot falls back to 32,768, the bench is a different server. |
| `block_size` | 64 | The hybrid block rounds up to 832. Prefill chunks and prefix-cache resumes then land on the Gated DeltaNet 64-token grid. |
| `mamba_ssm_cache_dtype` | float32 | bf16 state rounds every plain token and carries fp32 across a verify. The two paths then disagree. |
| `enable_prefix_caching` | on | |
| `async_scheduling` | off | It moves the recurrent state before acceptance is known. A cold start did not match itself. |
| `enforce_eager` | off | The decode step is a CUDA graph replay. |
| Packed recurrent Gated DeltaNet decode | off | With it off, a one-token decode and an 8-token verify use the same per-token kernels. |
| Batch-invariant mode | on | Norms, attention, and the unquantized bf16 products use one plan. The plan is not retuned per batch shape. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Otherwise warmup holds a reserved block it cannot use and the window fails to start. |
| Seed | 0 | |
| Clocks | not locked | Under load the card sits near SM 2,920–2,940 of 3,090 and memory 13,801 MHz. |

### Kernel rules that keep the greedy token

bf16 logits depend on the shape of the pass: how many rows, who else is in the batch, the order of a reduction, which processors a matrix is split across. A near-tie flips. Early speculative runs matched plain greedy on 9 of 16 prompts, and 7 of 16 with four requests in flight. Each rule below was one measured break, then a re-check.

1. **Gated DeltaNet uses the per-token kernels for both decode and verify.** Packed recurrent decode stays off. The backend is then batch-invariant. First fix: 11 of 16 prompts matched.

2. **Recurrent state is fp32.** Plain decode and verify round at the same time.

3. **One RMSNorm kernel for every batch shape.** Both norm classes (the decoder norm and the gated norm after each DeltaNet layer) go through one batch-invariant norm. A compiled norm whose tile depends on the batch writes a different sum for 1 row and for 8 rows. The check is bitwise across batch shapes, and within one bf16 unit of the native norm on the residual form.

4. **Async scheduling stays off.** Cold and warm then match. With it on, the same process matched 14 of 16 of its own outputs on a cold start and 16 of 16 once warm.

5. **Marlin is row-invariant only up to 16 rows.** Past 16 rows the K split follows the total row count, so a token’s bits depend on how many other rows share the call. Every call with M > 16 is executed as 16-row calls, same weights, fp32 split-K reduction left on. After that, plain-at-4 matched plain-at-1, and speculative-at-4 matched plain-at-1, 16/16. A K = 7 verify is 8 rows, so the usual one-request step does not take the split.

6. **Atomic split-K reduction only on those 16-row calls, and only on SM 10 or newer, bf16, K ≥ 2048.** The stock gate turns atomic reduction off once N ≥ 2048, which is every trunk linear. Forcing it on every row, including the 8-row verify, measured **181.6 / 462.2** tok/s and was not kept. Forcing it only when M ≥ 16 measured **193.1 / 491.6** against **193.0** one-request with the flag off. One request did not move. Four-wide did. On the fused gate+up (K 5120, N 34816) the atomic result matched the stock fp32 reduction at M = 8 and M = 16. Do not unchunk the wide calls to get the faster single kernel: at M = 228 the wide kernel differs by about one bf16 unit on the large layers, and that is enough to flip a later argmax. The one shape that stayed bitwise identical through M = 64 was N = K = 5120. The gate, up, and down projections did not.

7. **Attention follows the token, not the batch.** Decode and verify rows (query length ≤ 1 + K) use a split-KV kernel with a fixed segment count (16) and a tile plan taken from the server-wide max query length, never from the longest request in this batch. Prefill rows stay on FlashAttention. A mixed batch runs each row on its own path. KV positions are striped across segments (`tile t` belongs to segment `t % 16`), so the grouping is a function of position, not of the current length. A token at cache length *p* then matches whether it is alone or first in a verify block. Check: 57 of 57 positions around every alignment boundary, identical at query lengths 1, 2, and 3. Single-split attention left a 16k verify on too few processors, about 2.1 ms per layer.

8. **Gated DeltaNet prefill is a 64-token grid.** The KV block is 64 so a prefix-cache resume lands on that grid. Eight long prompts matched the uncached single-request run, including with other requests and with the cache warm.

9. **The bf16 matrix tile is chosen by `(N, K)`, never by `M`.** This is the unquantized path: the vocabulary projection, and the small DeltaNet projections that are not Marlin. The default tile (128×128×64, 8 warps) spent about 10 ms of a verify step on many small products and the vocabulary projection. Per weight shape, measured 16 Oct on this 5090:

| Shape | Tile | Warps | Stages |
| --- | --- | ---: | ---: |
| N ≤ 2048 | 16×64×256 | 4 | 3 |
| N ≥ 65536 (the 248,320-row head) | 16×256×64 | 8 | 3 |

The small tile went from about 150 µs to 21 µs. The head went to **1.53 ms**, about 1.3×, near the HBM ceiling. Both matched the default kernel bit for bit, including across batch sizes. That was the jump from about 177 tok/s to about 217 tok/s on the 32,768-window day. A tile that depended on `M` makes a 1-row decode and an 8-row verify disagree.

Greedy verify is the argmax of the target’s fp32 logits. The drafter may be wrong. It may not replace a token the target would not have chosen.

### CUDA graph

Capture size 64 covers one request (8 tokens) and four requests (32 tokens) and the pad up to 64. A FULL graph replay does not re-enter Python. Piecewise and eager do. Anything slow in Python therefore shows up on prefill and on batches wider than the capture, and it does not show up on the compiled decode step.

Prefill stays chunked at 8,192 tokens. That is what keeps the profile peak inside the 57,344 pool. Do not raise it to “speed up” a long prompt. A 16k prompt measured **75–79 s** of prefill at this cap.

On 3 Nov, graphs wide enough for one request only: **180.8 tok/s** one request, **67 tok/s** four-at-once. The same evening, after the capture held four requests: **210.9** and **350.5**, acceptance unchanged at 5.49 tokens per step.

### Bench

Sixteen prompts, temperature 0. Do not set a 384-token cut. The only ceiling is the server hard stop at 32,768 new tokens, and a reply that hits it is a truncated run, not a finished one. Report completion tokens divided by wall seconds. The wall includes prefill. Run thinking off and thinking on as two separate reads. Thinking on writes the reasoning and the answer in one generate; the token count includes both. Do not stop that generate at the close of the reasoning block and start a second request for the answer. A second request is only the repair when the block closed at once, or never closed and room remains to close it. Run one request, then the same sixteen prompts four at a time. Warm with one short call first, or the first request pays the clock ramp (idle memory clock is a few hundred MHz; under load it is 13,801).

A run is the same server only when the health check says window 57,344 and K = 7. A boot that falls back to 32,768 is not this number.

Identity against plain greedy needs a second process with the drafter off and the same kernels. That comparison was last done on 16 Oct: 16 of 16, including with four requests in flight. The 4 Nov strings match the 3 Nov speculative strings, 16 of 16. That is not a new plain-versus-speculative proof.

Draft counters, if the server exposes them: accepted draft tokens, draft tokens proposed, and the per-position accepts. Tokens per step is 1 plus accepted/drafts.

## Numbers

### 4 Nov 2026, window 57,344, K = 7

Sixteen prompts. Temperature 0. Wall includes prefill. The generation ceiling is the server hard stop at 32,768 new tokens, which none of the thinking-off replies reached. Every thinking-off reply stopped on its own.

| Read | Thinking off | Thinking on |
| --- | --- | --- |
| 1 request | **290.5 tok/s** (7,806 tokens, 26.87 s). Per request 282.8. | **228.7 tok/s** (63,219 tokens, 276.43 s). Per request 254.3. |
| 4 requests | **614.3 tok/s** (7,806 tokens, 12.71 s). Per request 173.3. | **467.4 tok/s** (63,219 tokens, 135.26 s). Per request 155.4. |
| How it ended | 16 of 16 stopped | 15 of 16 stopped. One reply was cut by the repetition guard at 4,675 tokens, both times. |
| Tokens kept per step | **5.67** across this run | **4.74** across this run |
| Acceptance | 0.666 | 0.534 |
| Per position | 0.885, 0.797, 0.714, 0.648, 0.588, 0.538, 0.494 | 0.835, 0.686, 0.573, 0.495, 0.434, 0.381, 0.332 |

Thinking on writes the reasoning and the answer in one generate. The token count is both. That is why the same prompts emit 63,219 tokens with thinking on and 7,806 with it off. The rate is still completion tokens divided by wall seconds. The non-thinking path is a separate generate and was not given the thinking repair.

Stopping the reasoning at the closing tag and prefilling it again for the answer was the previous thinking path. Same prompts, same 4.74 tokens kept per step: **180.3** one request and **352.1** at four. The drafter did not change. The wall did, because the answer now continues in the cache the reasoning just filled. An earlier boot of the non-thinking path on these prompts was 253.3 / 537.8; this boot measured 290.5 / 614.3 with the same 5.67 tokens per step.

An earlier pass of this bench stopped every reply at 384 new tokens (thinking off 235.1 / 510.8, thinking on 211.8 / 427.8). That cut the reply. Those rates are not this measurement.

How the same prompts moved while replies were still cut at 384 tokens. Those rates are a history of the kernels, not the measurement above:

| Change | 1 request | 4 requests |
| --- | ---: | ---: |
| 3 Nov, graph holds one request | 180.8 | 67 |
| 3 Nov, graph holds four requests at K = 7 | **210.9** (22.6 s) | **350.5** (13.6 s) |
| 4 Nov, atomic reduction only on 16-row calls | 193.1 | **491.6** |
| 4 Nov, Python scan removed from prefill | **235.1** | **510.8** |

The one-request 193 to 235 move is prefill. Decode is a graph replay and does not run that Python. Four-wide moved on the 16-row atomic reduction, then again when prefill got cheaper. Atomic reduction on every row, including the 8-row verify, was **181.6 / 462.2** and was not kept. Graphs captured through 256 tokens, so the short prefills (the longest here is 228 tokens) were graphed, measured **238.5 / 515.7**, same 16 strings, and cost extra graph memory (about 0.26 GiB to about 0.78 GiB). Not kept.

Thinking on and thinking off are the table at the top of this section. Serving keeps thinking on. A thinking-on reply is the reasoning and then the answer. A thinking-off reply is the answer. Do not score the two strings against each other as a greedy match.

A different thinking-on read, 3 Nov, eight reasoning prompts, max 1,536 new tokens, prompts about 285 tokens: **184.9 tok/s** (11,092 tokens in 60.0 s), **5.41 tokens per step**. That is not this sixteen-prompt bench.

### Decode only, evening of 4 Nov, same server

Prefill is a few dozen tokens. Thinking off. Temperature 0. Clocks sampled during the run.

| Run | Result | Card |
| --- | --- | --- |
| 900 tokens, easy text | **381 tok/s** (2.36 s) | SM 100%, memory clock 13,801 MHz |
| 700 tokens, easy text | **378 tok/s** (1.85 s) | SM clock 2,917–2,940 of 3,090. Memory bus about 68% busy. |

Eight tokens in a 21 ms step is 381 tok/s. That is the K = 7 ceiling for this step. The step stays about 23–24 ms when the prompt gets harder. The keep rate does not:

| Prompt | Kept per step | Wall |
| --- | ---: | --- |
| Short suite item | **6.42** | 286 tok/s over 77 tokens |
| Long suite item, 228 prompt tokens | **3.87** | 162 tok/s over 256 tokens |

### Dated, not re-run on 4 Nov

| Read | Date | Window | Result |
| --- | --- | ---: | --- |
| Reasoning, thinking on, max 1,536 new tokens, prompts about 285 tokens | 3 Nov | 57,344 | **184.9 tok/s** (11,092 tokens in 60.0 s). **5.41 tokens per step.** Per position 0.880, 0.784, 0.687, 0.609, 0.537, 0.483, 0.431. |
| 16k context, 4 prompts, 16.5k–18.0k tokens | 3 Nov | 57,344 | Prefill **75–79 s**. Three answers ran 256 tokens: **32.8, 55.9, 37.2 tok/s**. The fourth stopped at 18 tokens and is not a decode sample. **3.82 tokens per step**, acceptance 0.403. |
| Clean short bench | 23 Oct | 49,152 | **299.5 tok/s**. A second checkpoint the same day: 293.7. Thinking on **258–261**. Acceptance **5.58**. |

The keep rate did not fall from 23 Oct (5.58) to 3 Nov (5.49). The wall did. The window grew from 49,152 to 57,344.

### 16 Oct 2026, window 32,768

Same kind of 5090. This is the day plain and speculative ran under the same kernels. Do not quote it as the current server.

| Setup | 1 request | 4 concurrent | Tokens per step | Greedy identity vs plain |
| --- | ---: | ---: | ---: | --- |
| Plain, batch-invariant | 38.0 | 132.8 | 1 | the reference |
| MTP, 2 draft tokens | ~90 | ~254 | ~1.9 | 16/16 |
| DFlash, first measure | 177 | 440 | 5.69 | 16/16 at both concurrencies |
| DFlash, after the matmul tile tune | **217.5** | KV pool retuned | 5.35 on reasoning | 16/16 |
| DFlash, thinking on, ~1.4k tokens | **208.7** | — | 5.35 | same stack |
| DFlash, ~16k document question | 133.5 | — | ~3.0 | 4/4 |
| DFlash, a bit less GPU reserved | **226.6** | — | 5.70 | 16/16 |

Plain at 4 matched plain at 1 on 16/16. DFlash at 4 matched plain at 1 on 16/16. Eight prompts of about 3k tokens matched the uncached single-request run, cold and warm, 8/8.

## Where a K = 7 step goes

Measured 4 Nov on the real shapes, M = 8, memory clock at 13,801 MHz. The int4 chain is one CUDA graph of every trunk linear.

| Piece | Time |
| --- | ---: |
| 68 layers of int4, 12.59 GB | **10.67 ms**, 1.18 TB/s of weight traffic. Alone, the memory controller was about 90% busy. |
| Vocabulary projection, bf16, the 16×256×64 tile | **1.53 ms** |
| Gated DeltaNet recurrence, 48 layers, T = 8 inside one kernel | **1.17 ms** (about 25 µs a layer) |
| Gated DeltaNet convolution, 48 layers, width 10,240, kernel 4 | **0.65 ms** |
| Norms, 136 fused residual calls, 8×5120 | **0.63 ms** |
| Split-KV attention, 20 layers, context 64 to 1,024 | **0.41 ms**, flat across that range. At 16,384 tokens, **2.78 ms**. |

A pure read of 12.59 GB at 1.79 TB/s is 7.0 ms. The Marlin chain is already close to the bus when it runs alone. On a full serving step the SMs are at 100% and the bus is about 68% busy, because the rest of the step is not a weight read.

The widest layer, gate+up, K = 5120, N = 34816, int4 weight 89 MB, HBM floor 50 µs:

| Kernel | M = 8, cold | M = 228 | M = 4096 |
| --- | ---: | ---: | ---: |
| Marlin, fp32 reduce | 53 µs | 0.456 ms (178 TFLOP/s) | — |
| FP4 × FP4, block scale 16, both sides FP4 | 81 µs | 0.098 ms (832 TFLOP/s) | 1.23 ms (1,188 TFLOP/s) |
| FP8 tensor-core | 144 µs | 0.271 ms | 3.90 ms (375 TFLOP/s) |
| bf16 | 277 µs | 0.562 ms | 6.79 ms (215 TFLOP/s) |

FP4 at 8 rows is slower than Marlin and about 13% off the bf16 product (median relative). It is not the decode kernel. It is the wide product. Training-sized bf16 multiplies are the 6.79 ms row. Swapping that for FP4 changes the values.

## Published numbers on one RTX 5090

llama.cpp fork, LukasParke, DFlash2, n = 7, rejection sampling, IQ4_XS, fp16 KV, context about 131k. Plus a published plain decode of this model family at about **115 tok/s** on a 5090 with 32 GB. Their short numbers are often decode-only. Ours on the 16-prompt bench include prefill. Context and quant differ.

| Published workload | Published tok/s | Closest read here |
| --- | ---: | --- |
| Plain decode, this model family | ~115 | Batch-invariant plain, 16 Oct: **38** tok/s. Their llama.cpp plain band is ~70–85. The invariant kernels cost speed before the drafter is added. |
| Math / structured, lossless DFlash2 | 293–320 | **293.7–299.5**, 23 Oct, window 49,152. Current 16-prompt bench, window 57,344, replies not cut: thinking off **290.5**, thinking on **228.7**, prefill included. |
| Code | 212–258 | **258–261** thinking on, 23 Oct. **185** thinking on, 3 Nov, window 57,344. |
| Open prose | 114–127 | **290** thinking off, 4 Nov, replies not cut. **378** easy text, decode only, the same evening. **229** thinking on, same 16 prompts, 4 Nov. |
| Document copy, long context | up to ~380 | Not run as a copy task. 16k read, 3 Nov: prefill 75–79 s, then 33, 56, and 37 tok/s. |
| 4 streams, aggregate | ~285 | **614.3** thinking off, **467.4** thinking on, 4 Nov, prefill included, replies not cut. Not the same meter as a decode-only 285. 16 Oct, window 32,768, 4-wide aggregate was 440, with 16/16 identity. |
| MTP, lossless | ~130–160 | Invariant MTP, 2 drafts, about **90** tok/s on 16 Oct. Faster MTP runs were not batch-invariant. |

Same speculative idea on both sides: propose 7, verify, drop the rest. Theirs is IQ4_XS. This is W4A16, and a second request must not flip a token. Their context is about 131k. This window is 57,344.

## Measured and not kept

| Idea | What happened |
| --- | --- |
| Lookup drafting (copy a matching span out of the prompt) | 16 Oct, a 6-token match: short prompts 174 to 92 tok/s. 4 Nov, a 12-token match on the current drafter, K still 7: the three long reasoning prompts 224 / 231 / 201 tok/s fell to 151 / 166 / 144, and tokens kept per step fell from 4.74 to 4.44. Thinking off on the same boot fell from 290.5 / 614.3 to 239.8 / 503.9. The scan runs every step and the copies it takes are often wrong. Off. |
| K = 15 | 295 tok/s on easy text, 184 on reasoning. The drafter had not been trained for positions 8–15. Off. |
| Block-16 drafter at K = 15 | First boot needed 8.08 GiB of KV and had 6.45. The window fell to 32,768. Off. |
| Draft trees | Needs a Gated DeltaNet tree rule this stack does not have. Off. |
| Async scheduling | Cold start did not reproduce its own outputs. Off. With the GPU already at 100% SM during decode, there is no idle time for it to hide. |
| Atomic reduction on every row | 181.6 one request, 462 four. Slower on one request. Off. |
| Capture through 256 | 238.5 / 515.7, same strings, more graph memory. Off. |
| fp8 KV | Diverged from fp16 on a long prompt. Off. |
| FP4 tensor-core GEMM at decode width | 81 µs vs Marlin 53 µs on the widest layer, and the values differ. Off. |
| Unchunking Marlin past 16 rows | About 1.8–3× on the real shapes at M = 228, and not bit-identical on the large layers. Off. |

## What this is not

- 4-bit matching the official BF16, token for token.
- 217–227 tok/s, or 440 tok/s aggregate, on the 57,344 server. Those are 16 Oct, window 32,768.
- 299.5 tok/s on the 57,344 server. That is 23 Oct, window 49,152, same keep rate, lighter step.
- A new plain-versus-speculative identity run on 4 Nov. The 16/16 figure is 16 Oct. The 4 Nov check is 16/16 against the 3 Nov speculative strings.
- 33, 56, and 37 tok/s as a long-context generation speed without the 75–79 s prefill in front of them. One of the four 16k answers stopped at 18 tokens.