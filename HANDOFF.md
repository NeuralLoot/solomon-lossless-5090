# Handoff — method → tok/s (4 Nov 2026)

## Locked primary (window 57,344, K=7, replies not cut)

| | 1 request | 4 requests | Kept/step | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Thinking off | **290.5 tok/s** | **614.3 tok/s** | 5.67 | 0.666 |
| Thinking on | **228.7 tok/s** | **467.4 tok/s** | 4.74 | 0.534 |

Thinking on = reasoning + answer in **one generate** (answer continues in the cache the reasoning filled). Older stop+prefill path: **180.3 / 352.1**. Earlier non-thinking boot: **253.3 / 537.8**. Cut-at-384 deprecated: 235.1 / 510.8 off, 211.8 / 427.8 on.

Decode-only easy text (still tuning): 381 / 378 tok/s.

Full method text (weights, drafter taps, server settings, kernel rules 1–9, CUDA graph, bench method, where a K=7 step goes, measured-and-not-kept, public compares, what this is not): see **SOURCE_4NOV.md** (authoritative paste from Elliott).
