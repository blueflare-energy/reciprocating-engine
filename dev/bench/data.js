window.BENCHMARK_DATA = {
  "lastUpdate": 1788640149254,
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
          "id": "36f37bce1cf943052643e9477eccae552d036e0e",
          "message": "README: results with capacity buckets; no-scatter diagnostic; mme-bench shapes\n\nThe results table now comes from the published sweep after the bucket\nchange (six runner models) plus Qwen2.5-7B at batch 8 and 64 measured\nby hand (76% and 69% of the HBM ceiling).\n\nRENG_NO_SCATTER=1 builds the batched layer without its two ScatterND\nnodes (stale cache, wrong results) to time them: they cost 7% of a\n4-layer batch-64 step, so they overlap with the MME work despite their\n45 us TPC spans. reng-mme-bench takes m k n iters arguments so the\nprefill gemm shapes can be measured on their own.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T07:03:30-06:00",
          "tree_id": "b2960d6f58039863fa234830c300477ae1a83186",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/36f37bce1cf943052643e9477eccae552d036e0e"
        },
        "date": 1788616634206,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 35505.41402086608,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.202946478373544,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 742.5433765125015,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.33 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 30.001435023323094,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 43248.52876867638,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 13.646114724338632,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5699.499457044754,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.41 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.13951123679786,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24724.085006040015,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.367799840493383,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 406.5874381914035,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.46 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 51.30845267522997,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 27969.66711452732,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 27.56661165359726,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3173.305440639615,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.52 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 50.51657722695331,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 15087.870342386213,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 29.714409034532085,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 247.62874103137548,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.03 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 62.43571886776591,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 17161.854607214675,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 33.79896274408826,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1954.9056118879766,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.09 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.97710446651195,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 44570.74369419734,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.8645345298228198,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 751.1861757893494,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.26 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.359510785921318,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 51672.07869455741,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.480260273710366,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6737.820330388493,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.19 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.158358150071846,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14169.963397213298,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 15.578455445432889,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 361.9027057373058,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.77 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 51.02103376471578,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 17246.84677449943,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 18.961180528072955,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3044.616765524739,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.63 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 56.68337756098727,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 28873.543803421693,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.709787315598089,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 599.0710691675662,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.67 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 17.851609734737835,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 33233.61302144886,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.7230033355444,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5235.141920812016,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.55 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 20.585415655340174,
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
          "id": "bc5704e1ad2d2832749943e373a1b2922b801b79",
          "message": "Pick the card by module id; gemm benchmark on the graph runtime\n\nsynDeviceAcquireByDeviceType takes the lowest free module id and no\nenvironment variable steers it (HABANA_VISIBLE_DEVICES included, checked\nby reading which /dev/accel node the process opens), so when the lowest\nmodule was a card that faults on acquire every process in the machine\nfailed with \"scal_init fail\". Device acquisition now goes through one\nhelper: RENG_MODULE_ID names a module id (synDeviceAcquireByModuleId),\notherwise any free card. The bench workflow drops its dead\nHABANA_VISIBLE_DEVICES prefix and relies on the runner's RENG_MODULE_ID.\n\nreng-mme-bench is rebuilt on the graph runtime (Runtime::launch_only and\nprobe::bench_node time a compiled node over many launches; the old\nMatmulHpu path read back zeros) and takes the shape as arguments. The\nprefill gemm [1024 x 2048] x [2048 x 8192] runs at 82% of the MME peak\nstandalone against 44% inside the prefill recipe, m=64 and m=8 gemms\nmove 1.6-1.8 TB/s.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T07:56:02-06:00",
          "tree_id": "867a98ad0f114076fec43c08529bf3c0304a7ed9",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/bc5704e1ad2d2832749943e373a1b2922b801b79"
        },
        "date": 1788618116521,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 39056.51661030933,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 12.323418196443543,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 759.5074039095857,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.31 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 30.686843016695626,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 46630.84834628068,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 14.713330703805521,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5800.139824151995,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.38 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.65404959763095,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24843.707845759393,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.48569886138419,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 407.73526549391937,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.45 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 51.45330034463276,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 25716.04429277224,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 25.34546454853432,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3125.493890605638,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.56 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 49.755454194578306,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 14565.027956888429,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 28.68471086303954,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 246.31643205167632,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.06 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 62.10484065797943,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 16178.504007229008,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 31.862328793156163,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1937.275513943113,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.12 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.418170871232945,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 47112.17113496165,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.08489060391614,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 794.4626647422299,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.26 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.841109473221785,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 54125.38504872923,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.692975753240609,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6828.935330035776,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.17 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.295728806140886,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14245.854511984882,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 15.661890124622383,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 367.2563594914582,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.72 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 51.775791727629375,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 17220.57556074079,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 18.932297959955882,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3078.5935839656454,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.60 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 57.31594348843498,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 31001.518590012183,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.204297387800774,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 608.8421903969205,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.64 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 18.14277759750848,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 35568.64432265735,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.265630299281234,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5405.722146937453,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.48 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 21.25616439730157,
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
          "id": "e6e243a9fe0bea877be9595975207336a15e3e78",
          "message": "Prefill: ping-pong cache buffers and plain projection gemms\n\nThe 1024-token prefill layer of SmolLM2-1.7B spent a third of its time\nin two ScatterND nodes: with the output aliased to the input the kernel\nprocesses the block's rows serially (0.16 ms per layer and cache against\n0.014 ms for the same node writing a separate buffer). The wide recipes\nnow write the updated cache into a second buffer per layer and alternate\nbuffers per block: CachedModel flips after each wide launch and rebinds\nthe in-place decode recipe to the current buffer; BatchedModel's prefill\nalternates between the sequence's slot and the recipe's own buffers so\nthe last block lands in the slot. Copying the cache back with\nsynMemCopyAsyncMultiple was tried first and moved 200 MB in 3.3 ms.\n\nThe attention projections of models with hidden size 1024 and up are\nplain gemms over the natural weights (the buffers the batched decode\nrecipe binds by name) followed by a transpose into the head layout; the\nper-head batch_gemms ran each head as an N = head_dim gemm at a quarter\nof the MME rate. Smaller models keep the per-head form\n(RENG_HEAD_BLOCKS forces it).\n\nSmolLM2-1.7B prefill at 1024 tokens: 31.2k -> 35.2k tok/s; single\nsequence decode 2.70 -> 2.62 ms. Two probes, reng-attn-bench and\nreng-scatter-bench, time attention gemm orientations and cache-write\nkernels standalone.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T08:52:39-06:00",
          "tree_id": "f7b4dc211bef62ded556e422818caf0eefc881c2",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/e6e243a9fe0bea877be9595975207336a15e3e78"
        },
        "date": 1788621432431,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 35870.69957952814,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.318204240488608,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 732.7031734954336,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.35 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 29.60385527139072,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 46910.987089170965,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 14.801722275335269,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5835.299669763069,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.37 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.833809368465143,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 23185.3102221678,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 22.851199492135237,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 400.55861403615216,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.50 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 50.54765780111146,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 26163.96959746733,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 25.78693504847943,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3139.0539337777823,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.55 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 49.97131963234353,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 14504.806473585786,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 28.566109248171877,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 241.60598883260877,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.09 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 60.91717598164442,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 17436.499173256223,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 34.339854254240215,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1955.1389996143334,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.09 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.98450364497339,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 41943.87581729117,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.6367702886596662,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 729.8544963216531,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.29 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.122123024618812,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 54525.42381003854,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.7276613671256795,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6863.650946457249,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.17 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.348068234578792,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14863.382292695296,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.34080006591455,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 382.96831705506554,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.61 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 53.99086308425106,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 18581.54037195308,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 20.42854245120434,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3079.338210764486,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.60 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 57.3298066328736,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 29023.804735097696,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.744705748203861,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 591.6397841110125,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.69 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 17.630166224134793,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 35455.434578888984,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.23932201831958,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5443.206497691676,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.47 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 21.4035588619633,
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
          "id": "c8d3cb4bda5492952a0710ca0da7b8b350180d49",
          "message": "Plain projection gemms for every model size\n\nAn interleaved A/B (three repeats each) puts the plain-gemm projections\nahead for SmolLM2-1.7B (34.5k against 30.7k tok/s at 1024 tokens) and\nQwen2.5-0.5B (107k against 100k) and level for SmolLM2-135M, so the\nhidden-size threshold goes; RENG_HEAD_BLOCKS keeps the per-head form as\na diagnostic.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T09:01:46-06:00",
          "tree_id": "4c6a066a792a2b410c00fd7156923de4895efc52",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/c8d3cb4bda5492952a0710ca0da7b8b350180d49"
        },
        "date": 1788622885564,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 38740.26348221701,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 12.223631531075991,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 729.3214326156624,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.34 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 29.46722072251536,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 45908.20484079604,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 14.485316561789578,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5750.189099236247,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.39 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.398669327672412,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24572.482006896036,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.218381508633723,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 402.1094106783748,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.48 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 50.743357344806974,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 28060.77338725021,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 27.65640504402535,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3152.828716693043,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.54 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 50.1906035613381,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 15159.141376757958,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 29.854771896856448,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 243.70675754400264,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.07 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 61.446851996328704,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 16501.821420769436,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 32.49907776142757,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1937.0961617968044,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.12 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.41248480299494,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 43859.96472699399,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.8029059897949637,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 794.3426716340555,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.26 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.839774140231976,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 53951.67005181784,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.677913684532082,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6789.16782968736,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.18 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.235772842422124,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 15469.78286105174,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 17.007476751761452,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 383.2935172219916,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.60 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 54.03670979507689,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 18653.80991680255,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 20.50799557701274,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3078.5094713350245,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.60 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 57.31437751532036,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 29386.753553501338,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.829049720475374,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 603.2578790946035,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.66 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 17.976371721583323,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 35195.41117829631,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.17889640077385,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5342.528938663636,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.50 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 21.0076785914556,
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
          "id": "921a60a07ffb734b17769f38280ce9952b8b7dcb",
          "message": "Recipe cache on disk; bench workflow cleanup step and longer timeout\n\nRuntime::new_with looks a recipe up in $HOME/.cache/reng/recipes (or\nRENG_RECIPE_CACHE; 0 disables) before compiling and stores fresh\ncompiles there. The key is a digest of the graph's structure that the\nbuilder accumulates as tensors and nodes are declared (names, sizes,\ndtypes, persistence, aliasing, guids, operands, raw params) plus a\nsalt of the SynapseAI version and the compiler's environment knobs;\nthe compiled program depends on the structure, never on the weights.\nParam structs must carry no compiler-inserted padding since their raw\nbytes are hashed, so RmsNormParams gets an explicit zeroed pad. A file\nthat fails to load is compiled over. RENG_RECIPE_TRACE reports hits and\nthe launch-time split (graph build, compile, uploads).\n\nMeasured: a cache hit removes the 0.5 to 1.2 s compile per recipe; the\nrest of a launch is weight loading and host-side conversion, which is\nthe next target.\n\nThe bench job kills engine processes a cancelled job left behind, and\nits timeout grows to 120 minutes (a full run took 71).\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T11:03:40-06:00",
          "tree_id": "ea0f97247aa788599da6217f303dcc261efa288b",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/921a60a07ffb734b17769f38280ce9952b8b7dcb"
        },
        "date": 1788631855206,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 38694.9054831588,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 12.20931981975454,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 733.4963521335363,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.35 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 29.635902553914054,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 45483.131909832715,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 14.351194219428242,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5639.833864593934,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.43 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 28.83446230841839,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24990.71633935987,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.630588896588073,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 404.7206541211569,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.47 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 51.07287775793991,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 26058.76189732624,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 25.683243438533484,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3128.0377324439037,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.56 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 49.79595019632782,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 15473.038592901212,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 30.47296850536511,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 246.8061395927196,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.05 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 62.22831276478196,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 17563.400014908308,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 34.58977577597302,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1932.2274541473746,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.09 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.2581303416999,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 41742.692175364966,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.619326533707008,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 791.3205489316462,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.26 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.806142707517415,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 55226.38746995245,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.788438864725321,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6852.046464048262,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.17 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.330572593158223,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 15566.1064570885,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 17.11337489751796,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 383.737129321775,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.61 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 54.099250217026416,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 18487.01977416047,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 20.32462652142308,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3087.2543584162663,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.59 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 57.47718609660621,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 29971.358620418363,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.964903347911621,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 608.133656527232,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.64 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 18.121664125707717,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 37259.79801696369,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.658629568232723,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5426.675774437508,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.47 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 21.33855741321219,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 26006.613400517082,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 17.18065241479097,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 560.6270206771684,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.78 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 47.42320920416933,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 32637.322250497633,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 21.561072974006464,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4580.936632619332,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.75 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 48.95975834437401,
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
          "id": "3d86b8e175263cce92f59cef6f34e8a1ddc81c47",
          "message": "Weights bf16 end to end in the checkpoint's own layout\n\nThe loader keeps every projection as the bf16 [out, in] bytes the\nsafetensors file holds (f32 and f16 checkpoints are converted once);\nthe graph builder borrows those slices (Gb and Runtime carry the\nweights' lifetime; Cow covers the one scaled copy of wq) and declares\nthem as the gemms' transposed B operand, so nothing is transposed or\nconverted on the host and the upload is a memcpy into the pinned\nstaging buffer. The per-head diagnostic form (RENG_HEAD_BLOCKS) is a\nfree reshape of the same matrix. Tied embeddings are the head as they\nare. The CPU reference reads the bf16 matrices directly, which also\nmatches the device's rounding.\n\nLaunching Qwen2.5-3B goes from 60 s to 21 s wall and from 32 GB to 14 GB\nof host memory (graph build 6 s to 0.2 s per recipe; the safetensors\nread and the pinned uploads remain); SmolLM2-1.7B from 33 s to 15 s.\nThroughput and every correctness check are unchanged.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T12:35:13-06:00",
          "tree_id": "250a4511220411001f8c332cff2ea12b7c638e7f",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/3d86b8e175263cce92f59cef6f34e8a1ddc81c47"
        },
        "date": 1788634201110,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 36562.4968757632,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.536485544828567,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 737.0651911358606,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.36 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 29.780096542876432,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 43915.086577692586,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 13.856432267876132,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5740.364523461957,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.39 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.34843977008945,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24926.59313062041,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.56738973207656,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 405.2972759260815,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.47 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 51.14564334243775,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 25727.68586173967,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 25.356938357266102,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3104.4792105154147,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.56 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 49.420916681711745,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 14039.171702031304,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 27.649077105881968,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 243.14002824770458,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.06 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 61.30396005708766,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 16972.637688119892,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 33.42631446420205,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1942.3968407643783,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.12 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.58053421270506,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 43846.457925555565,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.801734874031126,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 791.9358361706497,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.26 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.812989878667077,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 51686.76528070759,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.481533683454032,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6728.295576337519,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.19 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.14399803683096,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14989.26101420853,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.47919111179457,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 384.7664956445725,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.60 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 54.2443702536526,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 18617.319328551835,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 20.46787783024755,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3090.5581589081467,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.59 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 57.53869484634048,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 28878.690098715488,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.710983239005335,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 600.3510684199382,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.66 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 17.889752199447162,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 35295.80782315035,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.202227105821924,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 4981.845357643078,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.50 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 19.589413041509808,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 25802.810087286067,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 17.0460145889549,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 562.4608315239891,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.78 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 47.578330509818514,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 31096.85217862299,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 20.5433979521699,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4587.572566975513,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.74 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 49.03068133862682,
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
          "id": "322641a6c9f6b4d71e8c004e266d525c692af4b3",
          "message": "Llama 3.1 rope scaling\n\nLlamaConfig reads the HF rope_scaling object and rope_caches_scaled\napplies the llama3 type as transformers does: inverse frequencies whose\nwavelength exceeds original_max_position_embeddings / low_freq_factor\nare divided by factor, those below original / high_freq_factor stay,\nand the band between is blended linearly. Other types are reported and\nignored; RENG_NO_ROPE_SCALING ignores every type.\n\nDeepSeek-R1-Distill-Llama-8B over a 1000-token prompt against the HF\nf32 reference: per-position argmax agreement 953 -> 993 of 1000 and\nlast-logits cosine 0.9936 -> 0.9998. The 8-token greedy check passes\neither way, so RoPE changes need reng-prefill --ref at long prompts.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T12:46:20-06:00",
          "tree_id": "56551acc84ef46d87e21910220411667e9506072",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/322641a6c9f6b4d71e8c004e266d525c692af4b3"
        },
        "date": 1788635011774,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 39588.44104121311,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 12.491255161432946,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 751.8875520210731,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.33 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 30.378973471896433,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 47084.78074585603,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 14.856559011867319,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5835.556391580058,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.37 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.83512189570148,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24451.587194550964,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.099228844813847,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 404.75923753706303,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.47 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 51.077746711535404,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 27754.71510327031,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 27.354757197331526,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3151.9522311789515,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.54 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 50.17665058738422,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 15463.33528437134,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 30.453858579835405,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 246.94593283174038,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.03 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 62.26355944630514,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 17448.63532416923,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 34.363755477153084,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1954.3984783664196,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.08 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.96102662262379,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 43407.649716544656,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.7636877306554783,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 794.9560140279428,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.26 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.846599668339941,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 51794.69634448815,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.4908919146220425,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6695.677958186485,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.19 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.094821681432153,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 15725.667692807438,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 17.288796494016548,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 384.77707191826767,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.58 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 54.24586129643496,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 18521.65704438849,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 20.362706730656697,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3081.5706187649353,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.59 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 57.37136865374679,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30573.87400363208,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.104919069683226,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 607.7538681919365,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.65 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 18.110346882241224,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 37144.20683316291,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.631767875600124,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5379.336974864291,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.49 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 21.152413679082372,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 25983.622198133242,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 17.165463822156028,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 548.0635881529581,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.78 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 46.36047361179914,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 29835.572778606125,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 19.710163626888818,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4536.027388675124,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.76 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 48.47978101500419,
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
          "id": "59f52a818a403f9b8ea067120487ea710215c2db",
          "message": "Phi-3: split the fused qkv_proj and gate_up_proj weights\n\nPhi-3 checkpoints store q/k/v as one [q + k + v, hidden] matrix and\ngate/up as one [2 * inter, hidden] matrix. In the [out, in] layout the\nparts are contiguous row blocks, so the loader slices them when the\nfused names are present and everything else stays as it is.\n\nPhi-3-mini-4k-instruct: 6 of 8 greedy tokens exact plus 2 near-ties\nagainst the HF f32 reference; batch 1 decode 205 tok/s (62.8% of the\nHBM ceiling), batch 8 1589 tok/s (64.0%). The 2047-token sliding window\nis not applied yet.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T12:54:48-06:00",
          "tree_id": "2ec99088af6c9af07c4570f080dfd17bd9994277",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/59f52a818a403f9b8ea067120487ea710215c2db"
        },
        "date": 1788635834759,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 36429.73770873457,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.494596332049323,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 735.9463568700477,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.36 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 29.73489159648623,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 46961.05777214449,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 14.817520969617734,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5842.056444032438,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.37 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.868354349341214,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24724.051576689468,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.367766892874787,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 404.97200071044745,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.47 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 51.104595915881674,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24913.13126869975,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.55412186163028,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3008.4472026682292,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.56 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 47.8921611202257,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 15363.099658987197,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 30.256452166278866,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 247.9571478907283,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.03 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 62.51852152734168,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 17539.948860771758,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 34.54359051784632,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1960.4564898430892,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.08 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 62.15308602839035,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 43257.02411274943,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.7506276423846634,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 795.1490639238099,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.26 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.848748007511684,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 54431.459682713496,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.719514148019269,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6786.059609807934,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.18 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.231086696280475,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 15611.812287472008,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 17.163623879979415,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 389.0118639136329,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.57 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 54.84288215855284,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 18659.573161899476,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 20.51433168773136,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3084.88794614499,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.59 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 57.43312924132099,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 28885.84584526755,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.712646129358371,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 599.3234499777211,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.67 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 17.859130384556035,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 37117.15278960595,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.625480913406289,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5364.569545368028,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.49 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 21.094345783514346,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 26073.83458107496,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 17.22506049362425,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 562.2734049160091,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.78 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 47.56247617010095,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 32686.431402671806,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 21.59351577080187,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4343.806074442893,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.74 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 46.42537383843803,
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
          "id": "9f29dc16d10ae84f9554c7bb583bf21a4bc4716b",
          "message": "Qwen3: per-head q/k RMSNorm and an explicit head_dim\n\nQwen3 normalizes every query head and every key head over head_dim\n(one shared gain each, q_norm and k_norm) after the projection and\nbefore RoPE, has no q/k/v biases, and states head_dim in its config so\nthat num_attention_heads * head_dim may differ from hidden_size.\nLayerWeights carries head_dim and the optional gains (empty when\nabsent); both layer builders add an rms_norm_fwd_bf16 over the head\nlayout before the rope nodes; the loader reads the gains and the\ndecoupled q width; the CPU reference does the same. With a q gain the\nattention scale folds into it, so wq uploads straight from the\ncheckpoint without a scaled copy.\n\nQwen3-1.7B: 7 of 8 greedy tokens exact plus a near-tie against the HF\nf32 reference, batch 1 decode 366 tok/s (51.6% of the HBM ceiling),\nbatch 8 2903 tok/s; Qwen3-0.6B (hidden 1024, 16 heads of 128): 7 of 8\nplus a near-tie, 567 tok/s. Every other listed model re-verified on\nthe merged tree.\n\nPatch written by a sub-agent against the previous layout and rebased\nby another onto the bf16 weights.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T13:18:04-06:00",
          "tree_id": "2d7914b84e4a504bb86c3295a3c995c83fc401aa",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/9f29dc16d10ae84f9554c7bb583bf21a4bc4716b"
        },
        "date": 1788637829156,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 7924.419855079696,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 37.9697315917881,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b1)",
            "value": 130.65185507414395,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.65 ms; ceiling 163 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b1)",
            "value": 80.15245141505949,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 8522.403989397231,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 40.834963052439456,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b8)",
            "value": 1025.1016028853055,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.80 ms; ceiling 1293 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b8)",
            "value": 79.2901017072542,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 24617.39661414635,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 24.262648876227814,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b1)",
            "value": 404.4321926205381,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.47 ms; ceiling 792 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b1)",
            "value": 51.03647596127112,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 27739.308149147757,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 27.339572264335285,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b8)",
            "value": 3138.9348860695995,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.54 ms; ceiling 6282 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b8)",
            "value": 49.969424484568876,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 29786.782487605906,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 26.70202925115231,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b1)",
            "value": 513.4406281445481,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.95 ms; ceiling 871 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b1)",
            "value": 58.963763828615264,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 33425.549549982425,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 29.96396143795819,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b8)",
            "value": 4021.6720997205016,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.99 ms; ceiling 6790 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b8)",
            "value": 59.23204289046608,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 9824.686151168455,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 23.481157096271026,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b1)",
            "value": 204.00439786305776,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.88 ms; ceiling 326 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b1)",
            "value": 62.5142174559234,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 11278.160418373052,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 26.954983850478293,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b8)",
            "value": 1589.2405534036072,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 5.03 ms; ceiling 2482 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b8)",
            "value": 64.0378266739085,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 37845.33498591296,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.941256678604944,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 733.1506207431657,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.36 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 29.621933756707296,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 46569.026940182084,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 14.693824329275333,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5831.06675289053,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.37 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.812167971759802,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 23604.35559371593,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 23.26420623173654,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 399.6742355224815,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.49 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 50.436055501449914,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 28375.59343161542,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 27.966688390203483,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3177.9284526898505,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.52 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 50.590172016243365,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 15430.616054085272,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 30.389420553132325,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 246.59989889635332,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.04 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 62.17631239485809,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 17448.72838557941,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 34.363938754497376,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1958.5361973905808,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.08 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 62.09220627787367,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 8911.012333467625,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 40.201599321840135,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b1)",
            "value": 136.66383439411374,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.31 ms; ceiling 173 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b1)",
            "value": 78.92892400357107,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 9238.130977427096,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 41.67737919544344,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b8)",
            "value": 1056.082184183563,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.56 ms; ceiling 1380 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b8)",
            "value": 76.54781063705529,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 44638.88709674607,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.8704429466519397,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 794.5498350657313,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.26 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.842079540673781,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 54108.55391243683,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.691516398920107,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6769.4953002778875,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.18 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.206113310160935,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 15500.962028455891,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 17.0417551233151,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 385.56953482083895,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.60 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 54.35758269522086,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 16899.629601047935,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 18.5794500242689,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3035.093073974513,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.62 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 56.50606952996392,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 31030.792000358797,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.21110009822542,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 611.7560841699519,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.63 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 18.229608187603443,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 36341.67262335309,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.445270717579731,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5427.866987151116,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.47 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 21.343241452196132,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 25433.128159519758,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 16.801792990118948,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 549.4194889480192,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.82 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 46.47516870263977,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 32569.015301903004,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 21.51594760826792,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4581.178218551943,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.75 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 48.962340346664945,
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
          "id": "abb061c253364baa0fecad043ffe7abeda0e627e",
          "message": "SmolLM3: NoPE layers\n\nSmolLM3 leaves every fourth layer without rotary position encoding\n(no_rope_layers in the config, 1 for RoPE and 0 for NoPE). LayerWeights\ncarries use_rope and both layer builders and the CPU reference skip the\nrotary nodes on a NoPE layer; everything else is the Llama path with\ntied embeddings.\n\nSmolLM3-3B: 7 of 8 greedy tokens exact plus a near-tie against the HF\nf32 reference; batch 1 decode 249 tok/s (62.7% of the HBM ceiling),\nbatch 8 2021 tok/s (64.3%).\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T13:24:10-06:00",
          "tree_id": "07366aa7241b97148ae0503f45262f44ed391196",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/abb061c253364baa0fecad043ffe7abeda0e627e"
        },
        "date": 1788640148445,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 7782.890907400561,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 37.2915978918933,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b1)",
            "value": 130.64420153613818,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.65 ms; ceiling 163 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b1)",
            "value": 80.14775611369686,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 8853.475984042889,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 42.42128924465849,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b8)",
            "value": 1030.8392846618028,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.75 ms; ceiling 1293 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b8)",
            "value": 79.73390295616633,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 24446.05338521499,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 24.093774780154966,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b1)",
            "value": 401.3929036193171,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.48 ms; ceiling 792 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b1)",
            "value": 50.65293923278989,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 28294.520426253635,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 27.886783683249085,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b8)",
            "value": 3176.756191172075,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.52 ms; ceiling 6282 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b8)",
            "value": 50.57151051623946,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 24370.68883942401,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 21.846832450997443,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b1)",
            "value": 449.3393728289536,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.15 ms; ceiling 871 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b1)",
            "value": 51.602345443776414,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 33211.385278015616,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 29.77197626275543,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b8)",
            "value": 4021.7208113872007,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.99 ms; ceiling 6790 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b8)",
            "value": 59.23276032626384,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 9817.641903659172,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 23.464321232015884,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b1)",
            "value": 204.84457286805082,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.87 ms; ceiling 326 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b1)",
            "value": 62.77167701813551,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 11452.283150766903,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 27.371139966863314,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b8)",
            "value": 1577.9255124236056,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 5.03 ms; ceiling 2482 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b8)",
            "value": 63.58189152203127,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 38999.04086733867,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 12.30528299959929,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 739.5646104559506,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.34 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 29.881082113146963,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 44133.03264831597,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 13.925200320064757,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5748.690213705987,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.39 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.3910060596818,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24408.201231889543,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 24.05646809335379,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 401.94518609437216,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.47 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 50.72263336638332,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 27844.178863065077,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 27.442931744180246,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3155.780843791449,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.54 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 50.23759914979942,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 15254.91741667031,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 30.043395497210987,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 246.25634562201233,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.06 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 62.08969080334364,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 17584.71198877752,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 34.631748081839405,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1964.7369171603136,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.07 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 62.288790018080526,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 9093.713416452716,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 41.025846383674775,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b1)",
            "value": 136.31269375411745,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.31 ms; ceiling 173 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b1)",
            "value": 78.72612599917068,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 9411.525525109588,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 42.45964027529212,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b8)",
            "value": 1054.6348126745286,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.53 ms; ceiling 1380 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b8)",
            "value": 76.44290107428232,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 45085.75804305267,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.909189174765726,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 801.7819403178079,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.24 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.922561402305652,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 51791.38772949742,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.490605039066066,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6710.2367229971815,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.19 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.116771383252294,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 15485.237789738781,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 17.02446789783673,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 383.61052790323043,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.58 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 54.08140195242983,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 17492.316534127065,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 19.23104994173088,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3045.4735115301746,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.62 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 56.699328093037764,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 29082.613936252274,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.758372142416985,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 574.697311191527,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.67 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 17.12530055782864,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 35187.78093310916,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.177123243910247,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5307.463649782699,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.51 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 20.869796265129537,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 25881.251560710236,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 17.097835088189065,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 555.1400823569395,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.80 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 46.95907134735271,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 31990.216991766207,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 21.13357823046454,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4524.078835254698,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.77 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 48.352078247001835,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      }
    ]
  }
}