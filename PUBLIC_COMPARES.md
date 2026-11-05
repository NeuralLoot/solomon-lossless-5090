# Public compares — one RTX 5090

Closest public: LukasParke llama.cpp fork, DFlash2 n=7, rejection sampling, **IQ4_XS**, fp16 KV, context ~131k. Often decode-only. Ours: W4A16, window 57,344, batch-invariant, 16-prompt bench with prefill included. Different meters.

| Published workload | Published tok/s | Closest read here |
| --- | ---: | --- |
| Plain decode, this model family | ~115 | Batch-invariant plain, 16 Oct: **38** tok/s. Their llama.cpp plain band ~70–85. |
| Math / structured, lossless DFlash2 | 293–320 | **293.7–299.5**, 23 Oct, window 49,152. Current 16-prompt (57,344, not cut): thinking off **290.5**, thinking on **228.7**, prefill included. |
| Code | 212–258 | **258–261** thinking on, 23 Oct. |
| Open prose | 114–127 | **290.5** thinking off, 4 Nov. **378–381** easy text decode-only. **228.7** thinking on. |
| Document copy, long context | up to ~380 | Not run as copy. 16k read, 3 Nov: prefill 75–79 s, then 33 / 56 / 37 tok/s. |
| 4 streams, aggregate | ~285 | **614.3** thinking off, **467.4** thinking on, 4 Nov, prefill included, not cut. |
| MTP, lossless | ~130–160 | Invariant MTP, 2 drafts, ~**90** tok/s on 16 Oct. |

Same speculative idea both sides: propose 7, verify, drop the rest. Theirs is IQ4_XS; this is W4A16 and a second request must not flip a token.
