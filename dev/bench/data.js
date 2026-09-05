window.BENCHMARK_DATA = {
  "lastUpdate": 1788612583093,
  "repoUrl": "https://github.com/blueflare-energy/reciprocating-engine",
  "entries": {
    "Gaudi2 inference": [
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "distinct": true,
          "id": "961f333cd0fdd09034e1d7a3a02ca6c720001449",
          "message": "Exact readback, compute-only KV cache, benchmark harness and workflow\n\nThe root cause of every readback race so far: on this stack\nsynStreamSynchronize returns before DMA copies have landed, host-to-device\nand device-to-host alike. A plain read of the pinned host buffer after the\nsync shows whatever was there before (zeros on a fresh buffer, the previous\nstep on a reused one), which is what the zeroed prefix rows, the \"stale\"\nmemcpy copies and the all-zero outputs were. runtime.rs now uses two\nsentinels: the device output is pre-filled with one NaN pattern (an element\nstill showing it was not written by the recipe) and the host buffer with\nanother before every copy (still showing it: the copy has not landed). The\nhost spins on the second and re-copies on the first; no timed window. Per-step\nuploads are fenced by reading the last one back until it is visible. A\n4-layer synthetic model that read back zeros in most runs now reads back in\n0.6 ms and is always right. The stability re-read stays as a diagnostic\n(RENG_STABILITY_MS), as do RENG_TPC_OUT, RENG_SETTLE_MS, RENG_STEP_TRACE,\nRENG_WS_SLACK_MB, RENG_PERSIST_ALL and RENG_DUMP_SCRATCH.\n\nThe KV cache no longer uses concat nodes or device-to-device copies. Each\nlayer and KV head keeps two cache buffers; the recipe reads one and writes\nthe other as cache_out = cache_in + gemm(place, block) for the rotated keys\nand the values, with a per-step 0/1 placement matrix that maps only the\nblock's real rows to their positions (padded rows are nonzero from the second\nlayer on, and placing them corrupted later positions). Attention runs over\nthe whole updated cache with a mask admitting positions up to each query's\nown. The buffers swap roles every launch (Runtime::rebind). Per-KV-head K and\nV projections replace the split. synLaunch rejects a tensor bound inside\nanother tensor's buffer, so in-place aliasing was not an option.\n\nreng-cache-test now takes a tail block size and prints per-block error.\nVerified: 2 layers with 40 single-token steps (top-1 304/304), SmolLM2\nshapes with 4 layers, rows 64; SmolLM2-135M teacher-forced 7/8 exact plus\none bf16 near-tie.\n\nreng-bench measures prefill and decode tok/s at batch 1 next to the\nreng-ceiling roofline and writes github-action-benchmark JSON. The bench\nworkflow runs it on a self-hosted runner labelled gaudi2 after every merge to\nmain and publishes to gh-pages (dev/bench). SmolLM2-135M: prefill 5739 tok/s\n(0.50%), decode 219 tok/s (2.4%, 4.4 ms/step of which 3 ms is the 30-layer\nrecipe on a 256-row padded block).\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T03:51:16-06:00",
          "tree_id": "63f6241641fb82f99a3a2b65b36365afc1b5be9f",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/961f333cd0fdd09034e1d7a3a02ca6c720001449"
        },
        "date": 1788601906762,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 5336.946668600754,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 0.46274333733667583,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 216.38617732402287,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.27 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 2.4080349739712275,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "5b647943a8d06202d61153e4192950ae5a571d90",
          "message": "reng-synapse: a narrow decode recipe sharing weights and cache with prefill\n\nRuntime::new_with(gb, out, parent) compiles a second recipe on the parent's\ndevice and stream and binds every persistent tensor that has the same name\nand shape in the parent (all weights, all KV cache buffers) to the parent's\nbuffer instead of allocating and uploading its own. CachedModel takes a\ndecode block size and routes blocks that fit through the narrow recipe; the\ncache buffers are now exactly `capacity` positions in both graphs so they can\nbe shared, with placement restricted to real rows (which never spill).\n\nSmolLM2-135M, decode blocks of 16 rows: 397 tok/s at 2.45 ms per step\n(4.4% of the HBM ceiling), up from 219 tok/s. The step is ~0.1 ms uploads,\n0.3 ms launch, ~1.9 ms recipe; a 1-row block is slower (4 ms), so 16 stays\nthe default. All cache tests, the teacher-forced SmolLM2 check and the\nsingle-recipe regression pass unchanged.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T04:03:56-06:00",
          "tree_id": "e92439eb30a0a4574aa06e06d5586814d77ddf42",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/5b647943a8d06202d61153e4192950ae5a571d90"
        },
        "date": 1788602667428,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 5103.0478790676525,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 0.4424629948134231,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 390.7794561524751,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.45 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 4.348755586709772,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "18095324337f924d74ca3e18ee9f46dfffc94f2c",
          "message": "reng-synapse: single-node probe runner and batched-attention contract tests\n\nprobe.rs: run_node builds a one-node graph over host inputs and reads the\noutput back through the full readback protocol, so a kernel's rank, layout,\nbroadcasting and parameter struct can be pinned before it goes into a model\ngraph.\n\nreng-bgemm-test: batch_gemm with the batch outermost, transpose_b, and a\nbatch-1 B broadcast all match the CPU reference (rel_L2 0.0027).\n\nreng-batched-test: everything a one-node-per-op attention layer needs:\n4-D batch_gemm with the inner batch dim of B broadcast (query heads of one\nGQA group against one K head), batch_gemm with A broadcast against per-head\nweight blocks, add_fwd_bf16 broadcasting a [n,m,1,1] mask over [n,m,g,h],\nsoftmax over the FCD of a 4-D tensor, rope_st2_fwd_bf16 on [hd,rows,heads]\nwith a 2-D table, and transpose (synTransposeParams is five u32 permutation\nentries plus the dim count). All six pass.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T04:10:15-06:00",
          "tree_id": "0e0682683be9a4c9b884869bb48d5a485047947b",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/18095324337f924d74ca3e18ee9f46dfffc94f2c"
        },
        "date": 1788603047355,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 5164.227268604694,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 0.44776759248856035,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 394.7591430751098,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.44 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 4.393043190537673,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "44b29de96bf743522f03ff07de548313bf2ab628",
          "message": "reng-synapse: batched attention, one node per step for all heads\n\nAttention now runs on 4-D tensors [.., heads-per-group, groups]: the Q/K/V\nprojections are batch_gemms of the normalised input (broadcast) against\nper-head weight blocks, RoPE runs once on Q and once on K with one 2-D\ntable, the KV cache is one [head_dim, capacity, 1, groups] tensor per layer\nupdated by one placement batch_gemm and one add for K and for V, scores and\ncontext are single batch_gemms whose size-1 dim broadcasts each KV head over\nits group, the mask add broadcasts over heads, and one transpose brings the\nper-head context back to [hidden, tokens]. A decoder layer is 28 nodes\ninstead of ~80.\n\nAll correctness checks unchanged: single- and multi-head layer tests,\n2- and 4-layer synthetic models, 40 single-token cache steps (top-1\n304/304), SmolLM2-135M teacher-forced 7/8 exact plus one bf16 near-tie.\nSmolLM2 decode: 613 tok/s at 1.56 ms per step (6.8% of the HBM ceiling),\nfrom 397; the 30-layer recipe compiles in 4.7 s instead of 7.\n\nreng-bench now warms the prefill recipe as well before measuring.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T04:14:11-06:00",
          "tree_id": "0162a53960d388a4e3336a15a2c37499ea72b01d",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/44b29de96bf743522f03ff07de548313bf2ab628"
        },
        "date": 1788603283239,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 5702.347442604091,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 0.4944256426185535,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 606.5587826683675,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.59 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 6.75003727362712,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "6192e24db1147bd365f9562c3382fe5137fc79d3",
          "message": "reng-model: one-row decode blocks by default\n\nWith attention batched over heads the narrowest decode recipe is the fastest:\nSmolLM2-135M 734 tok/s at 1.30 ms per step with 1-row blocks, against 613\nat 16 rows and 510 at 64.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T04:17:09-06:00",
          "tree_id": "62907b3372e65b46318a33282e8da6e4c7e4e264",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/6192e24db1147bd365f9562c3382fe5137fc79d3"
        },
        "date": 1788603464367,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 9423.349189345343,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 0.8170574531728997,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 683.2120411339765,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.40 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 7.603066471410127,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "ae6ea73790d8e1480a4e28366e2b9084063730c2",
          "message": "reng-batched-test: slice and masked-softmax probes\n\nslice (synSliceParams: axes/starts/ends/steps, five entries each) extracts a\nrange of a batch dim exactly; softmax_fwd_bf16 does not accept a mask as a\nsecond input (compile fails), so the mask stays a separate broadcast add.\nThe graph compiler's TPC fuser already merges the SwiGLU sigmoid and\nmultiplies into one fused kernel per layer.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T04:19:15-06:00",
          "tree_id": "e7b3d80d2225697230b89929a7266f89c91820f0",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/ae6ea73790d8e1480a4e28366e2b9084063730c2"
        },
        "date": 1788603582437,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 10279.753864143415,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 0.8913125623083056,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 721.8541129679579,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.30 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.033091446173637,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "8a9ba2a585c38f11d65ef9b6b070af7da74da2f6",
          "message": "reng-synapse: batched decode over a B-slot KV cache\n\nBatchedModel advances B independent sequences one token per launch of one\nrecipe. The layer is the batched-attention one with the sequence batch as a\nfifth, outermost dimension: the input [hidden, B] is read as\n[hidden, 1, 1, 1, B] against weights broadcast over sequences, every\nsequence has its own RoPE row, mask, placement column and cache slot\n([hd, capacity, 1, groups, B], slot b contiguous), and with one query row\nper sequence the context tensor already is [hidden, B] in memory, so no\ntranspose is needed. Prompts are prefilled one sequence at a time with the\nwide single-sequence recipe, which shares the weights (runtime sharing now\nkeys on name plus element count, so a 4-D weight and its 5-D view with a\ntrailing 1 are one buffer) and is bound per launch to that sequence's slot;\nthe first write buffer is chosen so the last prefill launch lands in the\nbuffer the next batched step reads.\n\nreng-batch-test prefills three synthetic sequences of 40, 300 and 256 tokens\n(one, two and exactly one launch) and advances them together; every\nsequence matches the CPU reference over itself (worst rel_L2 0.0071,\ntop-1 18/18) at 2 layers and at SmolLM2 shapes with 4 layers.\n\nreng-bench --batch B measures the batched decoder; the bench workflow records\nbatch 1 and batch 8 for every model the runner lists. SmolLM2-135M at\nbatch 8: 2540 tok/s aggregate, 2.43 ms per step. reng-batched-test gains\nthe two 5-D probes this rests on (batch_gemm with weights broadcast over the\nsequence batch; RoPE with a per-sequence table).\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T04:28:04-06:00",
          "tree_id": "bd6ebc74debe2a4234f3a49ea847e61538247d80",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/8a9ba2a585c38f11d65ef9b6b070af7da74da2f6"
        },
        "date": 1788604240060,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 9202.712902246627,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 0.797927044313783,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 728.8432685641449,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.31 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.110869663443694,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 3174.389363890997,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 0.27523743808331746,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 2302.368592267628,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.71 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 3.4711944823234506,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 6897.035185494414,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 7.582599356885997,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 291.4849392806129,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 3.37 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 41.09353893512284,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 2985.5539482222025,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 3.2823175232381105,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 719.9509508666408,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 10.55 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 13.40374001597279,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 8451.63476075502,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 1.9640357311131862,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 567.6883415841069,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.70 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 16.91644154841568,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 3112.614268735687,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 0.7233258196812452,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 2051.0306521163025,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.33 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 8.06498068902564,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "e50a1059023d9ecc55294669177c1b5d3ee1486a",
          "message": "Attention biases (Qwen2), exact fence, last-row readback, batched fixes\n\n- Qwen2-style attention biases: loaded when present, added after the\n  projections as broadcast adds (the Q bias scaled with the attention scale\n  like Wq). Qwen2.5-0.5B generates correctly against the HF reference.\n- reng-generate's near-tie rule now uses the reference's top-8 candidates\n  with logits (tools/oracle/generate.py): a mismatch passes when the\n  engine's token is within --margin of the reference's best logit.\n- Runtime::launch_and_read_range reads back a row range; generation and\n  batched prefill read only the last row, which takes a 1024-token prefill\n  from 162 ms to 14 ms (74k tok/s) and a 128-token one to 3.8 ms.\n- Mask and placement matrices are built as bf16 and uploaded as-is\n  (Runtime::upload_bf16); Runtime::fence replaces the whole-buffer\n  read-back with a 4 KB fence buffer written after the uploads and read\n  until visible, and also replaces the 20 ms settle after cache zeroing.\n- Batched decode projections are 2-D gemms with M = batch over the natural\n  [in, out] weights, whose [hidden, B] outputs are free reshapes of the head\n  layout (the per-head weight blocks were the wrong element order for a\n  plain gemm). Batch 64: 4868 tok/s at capacity 1024, 5864 at capacity 256\n  (the whole-cache add per step scales with capacity x batch; an in-place\n  scatter is next).\n- Probe runner accepts raw int32 inputs; reng-batched-test case 10 probes\n  ScatterND guids.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T04:43:38-06:00",
          "tree_id": "67f6ddd7cd7773fcbc719a6ead41b0fd6f8460e9",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/e50a1059023d9ecc55294669177c1b5d3ee1486a"
        },
        "date": 1788605161844,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 37640.349984679204,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.263630358704254,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 687.0585206174345,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.41 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 7.645871687701552,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 32436.126014096157,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 2.8123948268666243,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 2745.0326596694354,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.23 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 4.138582438121884,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 16749.408307523325,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 18.41429675869387,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 286.9123805812393,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 3.43 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 40.448899732117745,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 9057.51473206098,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 9.957830217649926,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 966.6487673019275,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.85 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 17.99665483888726,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 28396.6434280073,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.598962675867505,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 549.4421059291757,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.75 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 16.372725293694018,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 20711.690497280204,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 4.813092536523877,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 2191.821824749033,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.21 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 8.618594106405157,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "a302066b221f0c359ed9c8b76631d695327a59dd",
          "message": "reng-batched-test: complete the probe inputs (fixes --all-targets build)\n\nThe three single-input probes lacked the new raw field. The ScatterND probe\nnow also tries scatter_nd_update_fwd_bf16 and scatter_nd_fwd_bf16 with their\nparameter structs from perf_lib_layer_params.h.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T04:44:55-06:00",
          "tree_id": "3e8433e2f1d94ad2f36e6ced91b0f8dc28bdad6b",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/a302066b221f0c359ed9c8b76631d695327a59dd"
        },
        "date": 1788605300489,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 38667.66920654849,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.352704721773081,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 665.9986284174497,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.40 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 7.411508487645162,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 30131.393177634654,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 2.6125615081849456,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 2475.923218332664,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.88 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 3.7328562607206757,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 16761.135745033793,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 18.427189901578192,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 287.1585870044057,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 3.43 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 40.48360990706344,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 8767.202627852548,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 9.638661137681071,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 914.7021030169739,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 8.21 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 17.029533978869797,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 28537.966531653845,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.631804088570583,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 553.3257057961171,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.76 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 16.488451979155315,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 21696.10532238728,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 5.041856077007862,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 2252.1903122631957,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.20 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 8.855972658268847,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "6db21ec5c57fc650b01c98d93ceffc91fa3691b0",
          "message": "reng-synapse: KV cache updated in place by ScatterND\n\nThe cache update is now one scatter_nd_update_fwd_bf16 node per layer for\nkeys and one for values, with the node's output tensor in the same section\nas its input (Gb::scratch_alias; the runtime binds both names to one buffer),\nso only the written rows move. The per-step upload is an int32 index tensor\n(ONNX tuples (g, 0, position) for the wide recipe, (b, g, 0, position) for\nthe batched one); padded rows are sent to a trash slot at index capacity,\nwhich the mask never opens, so the cache has capacity + 1 positions. The\nplacement gemm, the whole-cache add, the ping-pong buffers and the cache\nzeroing on reset are gone; reset is free.\n\nVerified: cache tests (40 single steps 304/304, 12 blocks of 8), batch tests\n18/18 at both shapes, SmolLM2-135M and Qwen2.5-0.5B teacher-forced. Batch 64\ndecode of SmolLM2-135M at capacity 1024: 5715 tok/s (was 4361; 5864 was only\nreachable with a 256-position cache). The batch-64 recipe now spans 2.4 ms on\nthe device against a 6.7 ms host step: host-side logits handling is next.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T04:51:44-06:00",
          "tree_id": "0c2907e60a1bc169482ecd2cdd6490f7c119b148",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/6db21ec5c57fc650b01c98d93ceffc91fa3691b0"
        },
        "date": 1788605637516,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 40455.15845937907,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.507690108300304,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 707.8023161658498,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.38 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 7.87671723334236,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 41202.1930832945,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.572462217980633,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 3426.0429018460254,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.78 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 5.165315952029355,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 13905.853245229615,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 15.288092787462052,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 347.1965234257319,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.84 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 48.94775657619564,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14329.800211938305,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 15.754180013445755,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 1849.1533774056888,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.73 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 34.4267496147734,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 27957.279529548876,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.496860961825091,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 565.0565597778013,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.69 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 16.83801756146129,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 28045.97118117607,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.517471598438983,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 2964.165062264082,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.16 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 11.65557129123233,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "d160f7c4e3ce8c1fc60cf2f2b176277d7b0af4c5",
          "message": "reng-synapse: greedy token chosen on the device\n\nbuild_head can end the recipe with argmax_fwd_bf16 over the logits (which\nstay device-resident as a persistent scratch tensor), so the read-back\ntensor is int32 ids, [1, tokens]: four bytes per sequence per step instead\nof the logits row(s) and a host argmax. The runtime gains 4-byte outputs\n(int32 sentinels), launch_and_read_i32, and read_bf16_range to fetch any\npersistent tensor (the logits, for the tests and reng-prefill) after a\nlaunch. CachedModel::step_last_id, BatchedModel::step_ids/prefill_id,\nGenerator::feed_id and BatchedGenerator::step_ids/prefill_id use it;\nreng-generate and reng-bench decode through the id paths.\n\nAll correctness checks unchanged. SmolLM2-135M: 806 tok/s single, 5869 at\nbatch 8, 29519 at batch 64 (2.16 ms per step, the device span); SmolLM2-1.7B:\n371 (52% of the HBM ceiling), 2461 (46%), 6521 (22%).\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T05:01:39-06:00",
          "tree_id": "7bc48ee10c5beec7543e3d38c34627f7dfc40a2a",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/d160f7c4e3ce8c1fc60cf2f2b176277d7b0af4c5"
        },
        "date": 1788606237633,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 47088.981251963574,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.082879910439641,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 797.4329834828703,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.25 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.874164410000828,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 47043.683505717345,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.078952340693604,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 5891.590364626155,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.36 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 8.882529076570682,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 15043.629463653473,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.53896377623786,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 369.4449087359953,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.70 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 52.08433334152469,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14802.946329865328,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.27435677824638,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 2429.839960963862,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.30 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 45.237724983870386,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30744.804908600257,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.144640900360043,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 606.393996426558,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.65 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 18.069824311056877,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30337.78067224434,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.050054448595216,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 4648.265141256147,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.72 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 18.27772225784177,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "blueflare-ci",
            "email": "ci@blueflare.energy"
          },
          "committer": {
            "name": "blueflare-ci",
            "email": "ci@blueflare.energy"
          },
          "id": "8a12e45a6e247bf2d675a5a6a3ba50b4e03d251d",
          "message": "ci: update badges [skip ci]",
          "timestamp": "2026-09-05T11:15:09Z",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/8a12e45a6e247bf2d675a5a6a3ba50b4e03d251d"
        },
        "date": 1788607014864,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 39468.09343824446,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 12.453282142612043,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 760.3027135750315,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.31 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 30.718976400422456,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 39659.808695645435,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 12.513773643048806,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5264.507682469327,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.52 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 26.91555318597498,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24658.02877938712,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.302695513724455,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 406.70270200883493,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.46 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 51.32299815196225,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 22794.36168997932,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 22.46588470382351,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 2888.8182006108877,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.77 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 45.987759595046654,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 41090.78044874984,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.5628021150168943,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 789.1547060612038,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.27 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.782040311307782,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 43800.30057100792,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.797732771403481,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 5847.2667876457035,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.37 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 8.815704087571051,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14909.809629686923,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.39184227262317,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 370.856866760828,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.70 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 52.28339114605971,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14531.644116743257,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 15.97608716943421,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 2448.2198161704423,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.26 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 45.57991329603759,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30594.313664376554,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.109668946502369,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 602.0870991803872,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.66 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 17.941483864049324,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30442.692641894162,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.0744344487782,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 4627.618771270326,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.73 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 18.196537427638326,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "4c7a066117546233339b50f482a1a42981f1e335",
          "message": "reng-model: load sharded safetensors checkpoints\n\nLarger checkpoints ship as model-0000N-of-0000M.safetensors with a\nmodel.safetensors.index.json weight map; load_weights now follows the map\n(single-file checkpoints unchanged). reng-batch-test skips its CPU\nreference under RENG_NO_CHECK for profiling runs at large shapes.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T05:14:22-06:00",
          "tree_id": "8f092dc01cdaac07692b0e1c018fb5b309426f61",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/4c7a066117546233339b50f482a1a42981f1e335"
        },
        "date": 1788607973599,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 38956.21686438975,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 12.2917708345885,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 751.8413563988706,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.33 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 30.377107001866932,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 39956.977573053766,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 12.607538696082335,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 4841.733472974474,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.53 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 24.75405919495856,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24500.392963334012,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.147331300555933,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 402.0880659439004,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.46 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 50.74066379061476,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24700.05593211884,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.34411703631598,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 2877.9397944213406,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.74 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 45.814583751542074,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 47725.68469578977,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.138085685348974,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 806.3046980099141,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.24 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.972892522509696,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 47343.76730168498,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.10497129607191,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 5969.195546672336,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.34 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 8.999531489052819,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14910.615519527782,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.392728267782637,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 369.4212143918932,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.71 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 52.08099291352761,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14882.573155041271,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.361898496844514,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 2451.628953734113,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.26 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 45.643383166487965,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30660.21013741523,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.124982318561852,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 606.7770511685796,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.65 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 18.08123888958278,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 29222.112223926306,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.790789494701674,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 4547.535091205524,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.77 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 17.88163558423471,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "bfe-noah"
          },
          "distinct": true,
          "id": "bf0bf3a67b8eedaa6e6db68209f5d2b08d1ea379",
          "message": "Batched decode: capacity buckets for the KV cache\n\nAttention reads the whole K and V cache every step, so the batched step\ngrew linearly with the configured capacity: at batch 64 on SmolLM2-1.7B\na 1024-position cache holding 160-token sequences was 12.9 GB of reads\nper step against 3.4 GB of weights. The in-place ScatterND update was\nnot the cost (a windowed design with a flush recipe passed and was\nslower with the same slope).\n\nBatchedModel now compiles its decode and prefill recipes for the\nsmallest bucket of positions (256, doubling, clamped to the capacity)\nthat holds the longest live sequence and grows on demand: the next\nbucket's recipes are compiled as children of the first runtime (the\nweights are bound by name; per-step inputs carry a bucket suffix), the\nused rows of every sequence are copied with synMemCopyAsyncMultiple\n(Runtime::copy_d2d, fenced), and the old recipes are dropped.\nRENG_MIN_CAP overrides the floor; tiny values exercise growth in\nreng-batch-test (growth at prefill and mid-decode both pass).\nBatchedModel takes its ModelWeights by value so it can rebuild graphs;\nthe batched layer views carry no RoPE tables.\n\nRuntime::new_with frees the f32 host copies of the inputs after the\nupload; uploads validate against the tensor sizes instead.\n\nSmolLM2-1.7B b64 6506 -> 13752 tok/s (46% of the HBM ceiling), b8\n2449 -> 3083 (57%); SmolLM2-135M b64 29413 -> 40154; Qwen2.5-3B b64\n10730 -> 12895 (54%), b8 1818 -> 1940 (61%).\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T06:25:31-06:00",
          "tree_id": "7511aeb40f7d881ef2ae54c035fd4b4c509075f6",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/bf0bf3a67b8eedaa6e6db68209f5d2b08d1ea379"
        },
        "date": 1788612582271,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 38085.09340517927,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 12.016907133974971,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 752.7864390039732,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.31 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 30.41529175344579,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 45544.15592646664,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 14.370448995385608,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5682.248991533982,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.42 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.051315749215206,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24062.31236565992,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 23.71556364098358,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 407.84435598980104,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.45 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 51.46706679194397,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 27004.05989162933,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 26.614919264319163,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3148.3459924513236,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.54 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 50.11924204585211,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 14778.095352889371,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 29.104330836768273,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 246.13425292064343,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.04 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 62.05890703589111,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 16493.751799574628,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 32.483185258410586,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1943.8659207546914,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.11 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.627109005613534,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 29714.156776069707,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 2.57638476202137,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 804.6649849000844,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.24 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.954645116115357,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 53663.408012114516,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.652919742009743,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6780.6619547069795,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.18 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.222948854812845,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14728.187841466708,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.192167308306402,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 370.13201220880444,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.70 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 52.18120117072378,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 17216.01733868146,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 18.927286651368057,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3070.770574009907,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.61 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 57.1702980226392,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30916.373383804315,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.1845108929913,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 609.8552164647263,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.64 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 18.172964576890934,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 33605.03419164551,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.809316157880906,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5322.900780484882,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.50 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 20.93049753299143,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      }
    ]
  }
}