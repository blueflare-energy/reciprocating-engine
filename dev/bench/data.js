window.BENCHMARK_DATA = {
  "lastUpdate": 1788709377879,
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
      },
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
          "id": "7c0c4b8c181ed1f0b53947f10a598a60cd73e927",
          "message": "OLMo-2 post-norm layers and full-width q/k RMSNorm; RMSNorm epsilon honoured\n\nOLMo-2 (model_type olmo2), from a sub-agent patch: LayerWeights.post_norm\nmoves the two RMSNorms onto the branch outputs (g1 normalises the o_proj\noutput before the first residual add, g2 the down_proj output before the\nsecond; no input norm), and the q/k norms span the whole projection (one\nrms_norm over n_heads * head_dim before the head reshape, chosen from the\ngain length; the Qwen3 per-head form is unchanged). layer_cpu mirrors both;\nthe loader maps post_attention_layernorm / post_feedforward_layernorm and\naccepts q_norm / k_norm at either length. reng-layer-test uses the dense\ninput generator of the other tests; reng-norm-test takes [scale] [eps].\nOLMo-2-0425-1B: 8/8 exact; prefill at 257 tokens 252/257 argmax agreement,\nlast-logits cosine 1.0000; b1 505 tok/s, b8 4147.\n\nRMSNorm now receives ns_LayerNormKernel::ParamsRmsNorm (epsValid, eps, axis\nbitmaps, normalizedShapeDims, fastMath). The previous struct was the\nbackward kernel's ns_RmsNorm layout, and the forward kernel then applied a\nfixed epsilon of 1e-5 whatever the config said: reng-norm-test 256 256\n0.001 1e-6 went from rel_L2 0.54 to 0.0027. Every roster model re-verified\n(SmolLM2, Qwen2.5, Qwen3, TinyLlama, Phi-3 incl. the 300-token prompt,\nGranite, SmolLM3, Llama-3.2, the 8B distill at 1000 tokens).\n\nREADME: decode columns refreshed from the published sweep; OLMo-2 row.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T14:44:27-06:00",
          "tree_id": "f8e5843f0ffdaccf2eba6e558d84596e946bac4d",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/7c0c4b8c181ed1f0b53947f10a598a60cd73e927"
        },
        "date": 1788643283683,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 7687.346489546199,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 36.83379833465527,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b1)",
            "value": 129.9446468493622,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.69 ms; ceiling 163 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b1)",
            "value": 79.71859249399799,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 8729.135628970409,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 41.82551441262415,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b8)",
            "value": 1029.6495843041182,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.77 ms; ceiling 1293 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b8)",
            "value": 79.64188138279599,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 24268.172405647056,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 23.918457145367544,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b1)",
            "value": 399.20346931767045,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.50 ms; ceiling 792 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b1)",
            "value": 50.37664814334733,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 27518.144709484848,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 27.12159588913579,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b8)",
            "value": 3113.5039415926117,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.54 ms; ceiling 6282 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b8)",
            "value": 49.56458344589245,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 29069.384532998065,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 26.058925848607032,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b1)",
            "value": 500.74514006498794,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.98 ms; ceiling 871 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b1)",
            "value": 57.505807991506394,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 32010.12920526039,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 28.695123641758816,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b8)",
            "value": 3999.7127706266597,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.00 ms; ceiling 6790 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b8)",
            "value": 58.90862121647563,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 12863.345545137026,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 20.825892946549978,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b1)",
            "value": 269.60339466059327,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 3.71 ms; ceiling 482 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b1)",
            "value": 55.899663805727464,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 15044.05955484804,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 24.356492070543233,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b8)",
            "value": 2111.30346515207,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.79 ms; ceiling 3798 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b8)",
            "value": 55.59518065293874,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 7883.051967911544,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 37.771518018931125,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b1)",
            "value": 130.06113499263867,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.69 ms; ceiling 163 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b1)",
            "value": 79.79005577509032,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 8762.40711709404,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 41.98493421834076,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b8)",
            "value": 1030.5449124173492,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.76 ms; ceiling 1293 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b8)",
            "value": 79.71113369589318,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 26597.26442979517,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 20.99697917074165,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b1)",
            "value": 570.9952930627543,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.75 ms; ceiling 989 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b1)",
            "value": 57.72264859219037,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 32873.38418006909,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 25.951607343778797,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b8)",
            "value": 4552.719252957102,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.76 ms; ceiling 7811 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b8)",
            "value": 58.28508183229219,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 14247.259406085184,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 29.25699026608702,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b1)",
            "value": 257.2524314042722,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 3.89 ms; ceiling 380 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b1)",
            "value": 67.65738441927596,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 15332.596739813785,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 31.4857489980826,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b8)",
            "value": 2037.9664552950483,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.92 ms; ceiling 2989 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b8)",
            "value": 68.1811037291838,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 26949.85348122627,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 22.131386877874,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b1)",
            "value": 504.9334044910734,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.98 ms; ceiling 950 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b1)",
            "value": 53.162211505598954,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 30411.238045361628,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 24.973897356654355,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b8)",
            "value": 4186.116026708728,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.91 ms; ceiling 7234 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b8)",
            "value": 57.86921787834892,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 9845.404838047321,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 23.530675089410508,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b1)",
            "value": 202.8301666441874,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.91 ms; ceiling 326 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b1)",
            "value": 62.15439116527024,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 11359.75174505034,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 27.149988426698233,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b8)",
            "value": 1591.094653170546,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 5.03 ms; ceiling 2482 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b8)",
            "value": 64.11253689902644,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4714.5831855422,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 42.57075414849983,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b1)",
            "value": 72.02229582508234,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 13.85 ms; ceiling 87 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b1)",
            "value": 83.2617380809574,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4872.1557512099325,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 43.9935698438852,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b8)",
            "value": 572.2293322958577,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 13.93 ms; ceiling 687 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b8)",
            "value": 83.2842133459666,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 37853.885775308205,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.9439546906852,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 707.0954297428268,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.38 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 28.56920990980112,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 43169.028245781854,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 13.621030096330447,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5673.9505119809455,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.41 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.008888578196675,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 23853.681517571218,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 23.509939257086828,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 396.51658691458334,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.50 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 50.03758262957756,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 26082.22446734252,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 25.70636790238162,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3134.738352451074,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.55 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 49.90261890326215,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 14857.666457227739,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 29.261040053376355,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 244.22689967276497,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.08 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 61.577997709009004,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 16929.380395754357,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 33.34112252856718,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1946.3610863541417,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.11 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.706214174721424,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 8348.237328386655,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 37.6626672212033,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b1)",
            "value": 134.82779210840377,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.39 ms; ceiling 173 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b1)",
            "value": 77.86853489127506,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 9527.090043291228,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 42.98100398593484,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b8)",
            "value": 1058.879459755729,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.51 ms; ceiling 1380 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b8)",
            "value": 76.7505650476539,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 9488.591599453339,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 24.397475698176834,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b1)",
            "value": 195.9014892342444,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 5.10 ms; ceiling 304 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b1)",
            "value": 64.51265493101425,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 11900.260430924003,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 30.598462545502613,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b8)",
            "value": 1519.5745238819834,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 5.27 ms; ceiling 2386 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b8)",
            "value": 63.685781620164434,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 6857.573659716694,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 33.13998772061343,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b1)",
            "value": 124.6479467251092,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 8.02 ms; ceiling 162 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b1)",
            "value": 77.12801580232389,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 8012.053884692837,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 38.719141861406285,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b8)",
            "value": 978.9023523566511,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 8.16 ms; ceiling 1281 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b8)",
            "value": 76.44460168174464,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 45007.00069831174,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.9023604693638876,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 781.2869947784759,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.28 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.694485162599983,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 50722.151681226336,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.397896251045262,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6629.449043322596,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.21 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 9.994970838861642,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14194.956864520336,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 15.605933259309682,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 375.8415599031459,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.67 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 52.98613305179599,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 18040.280140181432,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 19.83348104076361,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3027.6936157141745,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.64 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 56.36830956914821,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30188.41580863563,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.0153442490415685,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 597.5936156310172,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.67 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 17.80758336575906,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 36759.73292905285,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.542421789697414,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 4938.577693868256,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.49 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 19.419277664720997,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 14821.839749279317,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 29.111187058860775,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b1)",
            "value": 246.64642087505547,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.05 ms; ceiling 398 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b1)",
            "value": 62.03101833521335,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 16627.559156317,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 32.65775322900473,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b8)",
            "value": 2028.708571987915,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.95 ms; ceiling 3144 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b8)",
            "value": 64.5339741889155,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 25013.06835114049,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 16.524290439131494,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 548.3904534544691,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.82 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 46.388123013278964,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 27716.00470977202,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 18.30992124625634,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4506.773672026764,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.78 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 48.1671255445967,
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
            "name": "Noah"
          },
          "distinct": true,
          "id": "95ced3067bf9bf316de54c91f5a6ca3335acc389",
          "message": "README and CHANGELOG: Gemma rows, decode columns from the CI run\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T16:38:51-06:00",
          "tree_id": "9b1f4dc8a6a6c18e41907868d95113e2ea3092fe",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/95ced3067bf9bf316de54c91f5a6ca3335acc389"
        },
        "date": 1788650244389,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 7899.415751805294,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 37.8499248289733,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b1)",
            "value": 130.04092469175626,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.69 ms; ceiling 163 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b1)",
            "value": 79.77765713629074,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 8767.020980739753,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 42.007041472554015,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b8)",
            "value": 1030.1563296032177,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.77 ms; ceiling 1293 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b8)",
            "value": 79.68107738657959,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 24472.17064903447,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 24.119515682476692,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b1)",
            "value": 398.9787639812445,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.50 ms; ceiling 792 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b1)",
            "value": 50.34829192267518,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 27508.177638570163,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 27.111772448188,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b8)",
            "value": 3160.8682144401155,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.53 ms; ceiling 6282 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b8)",
            "value": 50.31858617013608,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 29057.9745184265,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 26.048697468184518,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b1)",
            "value": 503.11802284868054,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.99 ms; ceiling 871 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b1)",
            "value": 57.7783109692241,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 32361.312026190415,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 29.009937568401533,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b8)",
            "value": 3929.8842335930553,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.00 ms; ceiling 6790 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b8)",
            "value": 57.88017165669168,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 12852.67282376636,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 20.808613697389333,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b1)",
            "value": 269.8074104165095,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 3.71 ms; ceiling 482 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b1)",
            "value": 55.9419645051721,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 15424.274726577241,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 24.972064468509323,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b8)",
            "value": 2128.2968960868184,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.76 ms; ceiling 3798 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b8)",
            "value": 56.04265439526147,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 7878.054607966461,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 37.747573248304064,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b1)",
            "value": 130.12097051823758,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.68 ms; ceiling 163 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b1)",
            "value": 79.82676374265604,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 8742.900139678928,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 41.89146684657548,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b8)",
            "value": 1030.5217475653812,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.76 ms; ceiling 1293 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b8)",
            "value": 79.7093419286543,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 26879.411609679864,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 21.21971780894859,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b1)",
            "value": 572.6654674813167,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.75 ms; ceiling 989 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b1)",
            "value": 57.89148867234799,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 28195.0273168238,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 22.25831919115151,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b8)",
            "value": 4486.342050144055,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.78 ms; ceiling 7811 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b8)",
            "value": 57.43530382429307,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 14188.285419973803,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 29.13588617943888,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b1)",
            "value": 257.16041270356686,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 3.89 ms; ceiling 380 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b1)",
            "value": 67.63318350279326,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 16647.486020704433,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 34.185896570012744,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b8)",
            "value": 2057.202409479036,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.89 ms; ceiling 2989 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b8)",
            "value": 68.82465141081549,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 8186.822784260398,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 37.184926014114836,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b1)",
            "value": 136.16630855920394,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.34 ms; ceiling 172 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b1)",
            "value": 79.18793740201573,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 8746.804565507395,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 39.72838904655703,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b8)",
            "value": 1073.2545580418853,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.45 ms; ceiling 1363 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b8)",
            "value": 78.73129375800018,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 26937.63684529968,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 22.12135450065808,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b1)",
            "value": 504.98115781164586,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.98 ms; ceiling 950 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b1)",
            "value": 53.167239242139615,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 30424.62435987957,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 24.984890281186274,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b8)",
            "value": 4190.7516887460615,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.91 ms; ceiling 7234 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b8)",
            "value": 57.93330165785652,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 9843.919276478086,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 23.52712457349105,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b1)",
            "value": 203.54229360475904,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.91 ms; ceiling 326 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b1)",
            "value": 62.37261224352016,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 11498.48387886172,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 27.48156044622157,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b8)",
            "value": 1589.576017480468,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 5.03 ms; ceiling 2482 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b8)",
            "value": 64.0513440677123,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4748.242612346574,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 42.8746849790476,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b1)",
            "value": 72.24054850180592,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 13.84 ms; ceiling 87 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b1)",
            "value": 83.51405018787717,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 5192.616825029817,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 46.8872020168387,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b8)",
            "value": 577.6568169417484,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 13.85 ms; ceiling 687 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b8)",
            "value": 84.07414801668112,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 37990.752219861984,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.987139863856822,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 727.3071338590711,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.37 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 29.38583577562207,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 44207.349912269565,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 13.948649485592075,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5744.115882511828,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.39 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.367619133121387,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 24050.44882271174,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 23.703871057015768,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 387.16577634569194,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.52 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 48.8575766173821,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 27001.987045638503,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 26.61287628896973,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3155.6220531682625,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.54 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 50.235071325444295,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 15103.423049239518,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 29.745038903595514,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 243.73871423105805,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.10 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 61.454909375777575,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 16877.05909597835,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 33.23807972216163,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1939.9903688571892,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.10 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.50424093292694,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 9136.01818067618,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 41.2167022726661,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b1)",
            "value": 136.57428347513203,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.32 ms; ceiling 173 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b1)",
            "value": 78.87720470482542,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 9812.54117506139,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 44.268802902146675,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b8)",
            "value": 1063.4085379866872,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.53 ms; ceiling 1380 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b8)",
            "value": 77.07884539171805,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 9484.094876514122,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 24.385913530337582,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b1)",
            "value": 196.13960962431568,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 5.10 ms; ceiling 304 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b1)",
            "value": 64.59107076448625,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 11959.141965199084,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 30.749861284344995,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b8)",
            "value": 1525.4135233688733,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 5.24 ms; ceiling 2386 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b8)",
            "value": 63.93049567686784,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 6851.105198345651,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 33.10872816132253,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b1)",
            "value": 124.67267335710882,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 8.02 ms; ceiling 162 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b1)",
            "value": 77.14331582220959,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 7871.269930858735,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 38.03878774013327,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b8)",
            "value": 980.2829487171487,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 8.16 ms; ceiling 1281 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b8)",
            "value": 76.5524154372305,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 44536.873922024984,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.8615978297988005,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 769.8288763028015,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.30 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.566974476073952,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 53599.88949210283,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.647412328546945,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6764.346472045849,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.18 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.198350615599558,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 15693.061374214412,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 17.252949112678856,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 379.6727144080782,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.61 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 53.526249111320226,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 18591.942840233733,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 20.439978923133726,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3091.3810798107165,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.59 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 57.55401563703946,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30419.81242383164,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.069117422299593,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 591.8005442789606,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.69 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 17.634956687114563,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 36550.59081852728,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.493820236319834,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5381.642997107619,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.49 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 21.161481327506692,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 14909.549123586416,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 29.283454742593744,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b1)",
            "value": 247.1550627577101,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.04 ms; ceiling 398 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b1)",
            "value": 62.158940621038816,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 15903.311592854101,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 31.235277567849533,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b8)",
            "value": 2012.2650383837197,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.97 ms; ceiling 3144 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b8)",
            "value": 64.01089926931387,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 22212.666148139742,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 14.674271133257008,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 487.2532531444413,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.05 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 41.21655237268153,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 32477.831105065496,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 21.455708930995314,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4591.959797678969,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.75 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 49.077570822649086,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noreply@anthropic.com",
            "name": "Claude Fable 5.1",
            "username": "claude"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "distinct": true,
          "id": "0d39bfb7f6378197dfa3c079188e2d1d3422e70e",
          "message": "Fused attention node (sdpa_recomp_fwd_bf16) behind RENG_SDPA\n\nOne sdpa_recomp_fwd_bf16 node per layer replaces the qk batch_gemm, mask\nadd, softmax_fwd_bf16 and av batch_gemm in the prefill, cached decode\nand batched decode recipes. The kernel takes the engine's own tensors:\nit broadcasts the size-1 K/V heads dim of the [hd, keys, 1, groups]\ncache over the query heads of a group, reads the additive\n[keys, queries, 1, 1] mask as it is, and accepts the batched recipe's\n5-D tensors with one mask row per sequence, so there is no grouped view,\nno mask tiling and no reshape; the attention scale stays folded into q.\nSoftcapped layers (Gemma-2) keep the four nodes.\n\nDefault per recipe: fused in the single-sequence decode recipe, the\nfour-node chain in prefill and batched decode; RENG_SDPA=1 fuses every\nrecipe and RENG_SDPA=0 none (read when a graph is built). Measured on\nQwen2.5-1.5B, SmolLM2-1.7B, Llama-3.2-3B and Qwen2.5-7B (three\nalternating repeats per cell, medians): a batch-1 decode step is never\nslower and Llama-3.2-3B's is 2% faster (the graph compiler expands the\nguid into the same MME gemms and TPC softmax pieces, 113 to 116 us per\nlayer against 118 to 120); prefill blocks and batch-8 decode are a wash\non three models and 5% and 2.5% slower on Qwen2.5-1.5B.\n\nEvery verified model agrees with its reference as before (cache and\nbatch tests, SmolLM2-135M, Qwen3-0.6B, Phi-3-mini, Gemma-3-270m,\nLlama-3.2-3B greedy, DeepSeek-R1-Distill-Llama-8B 1000-token prefill);\ntwo near-ties become exact matches. reng-sdpa-shapes probes the\nkernel's mask, batch and rank contracts at the target models' shapes;\nits timing loop is opt-in (--time) after it hung the kernel at one\nshape and raised a TPC illegal-instruction device error at another,\nboth of which run fine inside the engine.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T17:16:45-06:00",
          "tree_id": "63f83d3f576be491364690f4e21ddc304f6419cd",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/0d39bfb7f6378197dfa3c079188e2d1d3422e70e"
        },
        "date": 1788651617527,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 7335.564300032145,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 35.1482395733968,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b1)",
            "value": 129.4274609022442,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.72 ms; ceiling 163 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b1)",
            "value": 79.4013086599843,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 8495.637904761974,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 40.70671378398198,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b8)",
            "value": 1027.0584733194642,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.78 ms; ceiling 1293 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b8)",
            "value": 79.44146275801798,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 24406.753806404864,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 24.055041525920206,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b1)",
            "value": 399.3115444704813,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.50 ms; ceiling 792 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b1)",
            "value": 50.39028646156009,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 27481.253973912775,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 27.085236765626497,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b8)",
            "value": 3160.8614236371686,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.53 ms; ceiling 6282 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b8)",
            "value": 50.31847806578624,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 26024.347810528925,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 23.329236609158766,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b1)",
            "value": 510.62197833959124,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.96 ms; ceiling 871 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b1)",
            "value": 58.64006875599187,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 29051.408254248396,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 26.042811213829,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b8)",
            "value": 3951.9872452081654,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.02 ms; ceiling 6790 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b8)",
            "value": 58.20570951744508,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 12566.697995052058,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 20.345617414865966,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b1)",
            "value": 269.128174627012,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 3.71 ms; ceiling 482 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b1)",
            "value": 55.801131514825194,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 15792.807252303852,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 25.568722538618122,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b8)",
            "value": 2130.5880706043595,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.75 ms; ceiling 3798 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b8)",
            "value": 56.10298596924529,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 7705.3303488823085,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 36.91996771039056,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b1)",
            "value": 129.6243307850243,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.70 ms; ceiling 163 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b1)",
            "value": 79.52208462375204,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 8694.819388010552,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 41.661088690323524,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b8)",
            "value": 1028.4528015912028,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.78 ms; ceiling 1293 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b8)",
            "value": 79.54931199966221,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 23908.060060782507,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 18.8740101612381,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b1)",
            "value": 569.4672448847249,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.76 ms; ceiling 989 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b1)",
            "value": 57.56817623649157,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 29567.29192249304,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 23.34164155379664,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b8)",
            "value": 4422.10055303895,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.81 ms; ceiling 7811 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b8)",
            "value": 56.61286766959982,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 13281.314850170167,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 27.27340664031862,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b1)",
            "value": 256.60345845891317,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 3.83 ms; ceiling 380 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b1)",
            "value": 67.4867045473611,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 15528.786731039958,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 31.888628492307028,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b8)",
            "value": 2039.6887253345365,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.92 ms; ceiling 2989 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b8)",
            "value": 68.23872306433375,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 7352.532327016742,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 33.39554034590044,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b1)",
            "value": 132.67691622518095,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.53 ms; ceiling 172 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b1)",
            "value": 77.15867051036356,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 8681.407617674064,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 39.431353098569176,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b8)",
            "value": 1067.5785121633683,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.48 ms; ceiling 1363 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b8)",
            "value": 78.31491310338573,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 26068.86947255956,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 21.40791734426019,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b1)",
            "value": 499.5415223518527,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.00 ms; ceiling 950 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b1)",
            "value": 52.594524012260244,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 29845.813148304835,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 24.509566923249096,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b8)",
            "value": 4168.234164231194,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.92 ms; ceiling 7234 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b8)",
            "value": 57.62201751668175,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 9384.690563656979,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 22.4295606021892,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b1)",
            "value": 201.73134985396464,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.94 ms; ceiling 326 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b1)",
            "value": 61.81767454304176,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 11395.860475939655,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 27.23628887126339,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b8)",
            "value": 1580.3747971186137,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 5.06 ms; ceiling 2482 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b8)",
            "value": 63.680584491096326,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4328.5570974725115,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 39.08509676514581,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b1)",
            "value": 70.63271713063963,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 14.17 ms; ceiling 87 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b1)",
            "value": 81.65530862777028,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4999.017624690804,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 45.139080573939516,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b8)",
            "value": 573.6202964768602,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 13.94 ms; ceiling 687 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b8)",
            "value": 83.48665902826399,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 35014.99216129218,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.048204729934787,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 704.1823015309495,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.42 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 28.451509005681917,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 44474.149856429714,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 14.032832294813865,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5784.455680799816,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.38 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.573862157506632,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 23197.5251140394,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 22.863238361930737,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 391.45614904242694,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.54 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 49.398991240147446,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 26473.41076805983,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 26.091917032870928,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3132.834533861634,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.55 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 49.87231157842418,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 13153.027251428337,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 25.90388324675096,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 238.1674255077789,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.16 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 60.050195952735386,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 16405.249917080593,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 32.30888755588023,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1931.4499880185972,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.14 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.23348204196133,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 8464.104855446976,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 38.18539794168463,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b1)",
            "value": 136.36241210422486,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 7.33 ms; ceiling 173 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b1)",
            "value": 78.75484036895702,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 9758.001468293343,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 44.0227496641341,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b8)",
            "value": 1062.8016908942413,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.51 ms; ceiling 1380 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b8)",
            "value": 77.03485940557616,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 9329.352906607273,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 23.988034307618392,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b1)",
            "value": 194.63519352330957,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 5.14 ms; ceiling 304 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b1)",
            "value": 64.09564892172108,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 11871.710447869204,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 30.52505359850958,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b8)",
            "value": 1516.586240202616,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 5.27 ms; ceiling 2386 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b8)",
            "value": 63.56054183834898,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 6524.123737550201,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 31.528553870336186,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b1)",
            "value": 123.27909011633106,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 8.08 ms; ceiling 162 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b1)",
            "value": 76.28101272744942,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 7134.036585387858,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 34.47602556965139,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b8)",
            "value": 966.3539305017354,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 8.24 ms; ceiling 1281 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b8)",
            "value": 75.46466828172352,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 43937.04782170071,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.809589528288366,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 739.5751395052497,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.30 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 8.230298366708915,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 50030.29666695623,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.33790852432878,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6643.206632245341,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.20 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.015712637945896,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14812.136972459377,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.284460969213765,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 378.176287869303,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 2.64 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 53.31528293795128,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 18591.61845176736,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 20.439622290506758,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3071.4946531277105,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.61 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 57.1837786191081,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 28812.15152490563,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.695520652161579,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 584.3215214563776,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.71 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 17.412090647511835,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 34728.63186113892,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.070423746264604,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5318.399157698534,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.50 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 20.912796431935867,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 14061.417415043497,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 27.61766147835712,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b1)",
            "value": 242.85768728119493,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 4.12 ms; ceiling 398 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b1)",
            "value": 61.07816038497755,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 15872.62663542694,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 31.175010047037976,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b8)",
            "value": 2014.421178264534,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.97 ms; ceiling 3144 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b8)",
            "value": 64.0794868808306,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 23914.930109550933,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 15.79883145138937,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 541.8023038399044,
            "unit": "tok/s",
            "extra": "ctx ~160; median step 1.85 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 45.830834145784685,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 29349.218062597927,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 19.388864917312695,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4498.9307921919435,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.77 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 48.08330305757461,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noreply@anthropic.com",
            "name": "Claude Fable 5.1",
            "username": "claude"
          },
          "committer": {
            "email": "noah@it.bluefla.re",
            "name": "Noah"
          },
          "distinct": true,
          "id": "d9b1a42b89b18047c51c1fd1c4715cfa0dff407d",
          "message": "Qwen2.5-32B on the roster\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01VUoyvmdyW73XuUwedpKuqZ",
          "timestamp": "2026-09-05T17:50:53-06:00",
          "tree_id": "0f1d46f4981a79ea3c545ca0c63c993c2712c0ea",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/d9b1a42b89b18047c51c1fd1c4715cfa0dff407d"
        },
        "date": 1788654148074,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 7284.291913901035,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 34.90256875142226,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b1)",
            "value": 132.53943097773873,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.54 ms; ceiling 163 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b1)",
            "value": 81.31044366721129,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 8328.53805283823,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 39.90605750344619,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b8)",
            "value": 1027.9109173370543,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.77 ms; ceiling 1293 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b8)",
            "value": 79.50739804937272,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 24178.91794808635,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 23.830488884613715,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b1)",
            "value": 420.6102768576453,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.38 ms; ceiling 792 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b1)",
            "value": 53.078035516450335,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 25879.522130580186,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 25.506586597301244,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b8)",
            "value": 3125.1149792059,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.56 ms; ceiling 6282 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b8)",
            "value": 49.74942221708824,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 28122.022333719207,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 25.209673561385152,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b1)",
            "value": 545.8696343392463,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.83 ms; ceiling 871 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b1)",
            "value": 62.68792619062173,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 29267.15222320319,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 26.236212490794298,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b8)",
            "value": 3901.378039874522,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.05 ms; ceiling 6790 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b8)",
            "value": 57.460326366694645,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B prefill tok/s (b1)",
            "value": 16029.654861493764,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 59826 tok/s"
          },
          {
            "name": "Gemma-2-2B prefill % of ceiling (b1)",
            "value": 26.793606048741193,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B decode tok/s (b1)",
            "value": 291.72773196639446,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.43 ms; ceiling 467 tok/s"
          },
          {
            "name": "Gemma-2-2B decode % of ceiling (b1)",
            "value": 62.45639850788113,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B prefill tok/s (b1)",
            "value": 16467.03945964182,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 59826 tok/s"
          },
          {
            "name": "Gemma-2-2B prefill % of ceiling (b1)",
            "value": 27.524695439986754,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B decode tok/s (b8)",
            "value": 2178.5797106999325,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.67 ms; ceiling 3663 tok/s"
          },
          {
            "name": "Gemma-2-2B decode % of ceiling (b8)",
            "value": 59.476152309710876,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B prefill tok/s (b1)",
            "value": 26238.847721002334,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 156572 tok/s"
          },
          {
            "name": "Gemma-3-1B prefill % of ceiling (b1)",
            "value": 16.758317224213386,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B decode tok/s (b1)",
            "value": 504.79159221647956,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.98 ms; ceiling 1223 tok/s"
          },
          {
            "name": "Gemma-3-1B decode % of ceiling (b1)",
            "value": 41.28501290419543,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B prefill tok/s (b1)",
            "value": 26262.511239790576,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 156572 tok/s"
          },
          {
            "name": "Gemma-3-1B prefill % of ceiling (b1)",
            "value": 16.773430721525123,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B decode tok/s (b8)",
            "value": 3680.327183386807,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.17 ms; ceiling 9654 tok/s"
          },
          {
            "name": "Gemma-3-1B decode % of ceiling (b8)",
            "value": 38.12094553425092,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m prefill tok/s (b1)",
            "value": 48079.59577079856,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 582419 tok/s"
          },
          {
            "name": "Gemma-3-270m prefill % of ceiling (b1)",
            "value": 8.255152772762248,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m decode tok/s (b1)",
            "value": 1057.1976643464152,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 0.95 ms; ceiling 4545 tok/s"
          },
          {
            "name": "Gemma-3-270m decode % of ceiling (b1)",
            "value": 23.259798486703666,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m prefill tok/s (b1)",
            "value": 51076.09450234675,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 582419 tok/s"
          },
          {
            "name": "Gemma-3-270m prefill % of ceiling (b1)",
            "value": 8.769644511216972,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m decode tok/s (b8)",
            "value": 7249.862401293194,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.10 ms; ceiling 35168 tok/s"
          },
          {
            "name": "Gemma-3-270m decode % of ceiling (b8)",
            "value": 20.61469315175564,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 12612.557219624825,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 20.419863986118596,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b1)",
            "value": 281.5954159645426,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.55 ms; ceiling 482 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b1)",
            "value": 58.38609377106897,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 13196.044980368823,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 21.36453686287814,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b8)",
            "value": 1984.9026751820438,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.03 ms; ceiling 3798 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b8)",
            "value": 52.266774827321605,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 7549.229825214097,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 36.1720145360998,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b1)",
            "value": 132.54426501602833,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.54 ms; ceiling 163 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b1)",
            "value": 81.31340925862156,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 8398.04020578151,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 40.23907596291311,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b8)",
            "value": 1019.71773209391,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.83 ms; ceiling 1293 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b8)",
            "value": 78.87366721780755,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 24771.401571706727,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 19.555567611251846,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b1)",
            "value": 609.070411708663,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.64 ms; ceiling 989 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b1)",
            "value": 61.571711308478285,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 31566.690938698135,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 24.92004972460953,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b8)",
            "value": 4509.242732219219,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.77 ms; ceiling 7811 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b8)",
            "value": 57.72848424122726,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 12353.719415375672,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 25.368573589054172,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b1)",
            "value": 273.2686656984046,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.66 ms; ceiling 380 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b1)",
            "value": 71.86965372484534,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 15391.930020109165,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 31.60759090146757,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b8)",
            "value": 2045.014360833757,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.91 ms; ceiling 2989 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b8)",
            "value": 68.41689464583979,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 8124.888679503737,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 36.903618458813334,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b1)",
            "value": 138.7154999325474,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.21 ms; ceiling 172 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b1)",
            "value": 80.67042752041615,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 8625.68414879167,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 39.17825453735135,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b8)",
            "value": 1071.483377324981,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.47 ms; ceiling 1363 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b8)",
            "value": 78.6013643314012,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 26408.452520387633,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 21.686785049937555,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b1)",
            "value": 530.1001948176196,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.89 ms; ceiling 950 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b1)",
            "value": 55.81191187867173,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 30457.410219496203,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 25.011814232509607,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b8)",
            "value": 4177.272819466444,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.92 ms; ceiling 7234 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b8)",
            "value": 57.74696864221165,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 9793.474745497813,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 23.406561337335702,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b1)",
            "value": 210.13075156185428,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 4.76 ms; ceiling 326 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b1)",
            "value": 64.3915505494754,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 10128.872092721911,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 24.208166363558853,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b8)",
            "value": 1575.8739412143125,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 5.08 ms; ceiling 2482 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b8)",
            "value": 63.49922426235902,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4715.071194628119,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 42.575160670561544,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b1)",
            "value": 73.02100861310977,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 13.69 ms; ceiling 87 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b1)",
            "value": 84.41630503306897,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 5038.422027233212,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 45.494886181134135,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b8)",
            "value": 577.0634816874054,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 13.86 ms; ceiling 687 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b8)",
            "value": 83.98779197528401,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 36963.07879344425,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.662880290749502,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 800.8484689210043,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.25 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 32.35717139745678,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 43895.826630003365,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 13.850355218246131,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 5747.015222541267,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.38 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 29.382442426290172,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 22531.351583610765,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 22.206664691173895,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 420.8662044635097,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.38 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 53.110331766211985,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 25907.019555018094,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 25.533687771507214,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3123.310202252514,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.56 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 49.72069155877355,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 14751.567469290232,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 29.05208619480109,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 255.3019034962874,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.92 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 64.37038692160556,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 17301.094186075432,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 34.07318446698054,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 1950.0664313743932,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 4.10 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 61.82368611505865,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 9085.26370936128,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 40.9877259405461,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b1)",
            "value": 139.08641393259904,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.19 ms; ceiling 173 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b1)",
            "value": 80.32806224035063,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 9332.553882807675,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 42.10336354681252,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b8)",
            "value": 1062.9630994934375,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 7.52 ms; ceiling 1380 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b8)",
            "value": 77.04655875537252,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B prefill tok/s (b1)",
            "value": 24009.05441464612,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 259893 tok/s"
          },
          {
            "name": "Qwen3-0.6B prefill % of ceiling (b1)",
            "value": 9.238041284074358,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B decode tok/s (b1)",
            "value": 639.8350505239749,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.56 ms; ceiling 2024 tok/s"
          },
          {
            "name": "Qwen3-0.6B decode % of ceiling (b1)",
            "value": 31.608343724906085,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B prefill tok/s (b1)",
            "value": 29869.900830179056,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 259893 tok/s"
          },
          {
            "name": "Qwen3-0.6B prefill % of ceiling (b1)",
            "value": 11.493138057618427,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B decode tok/s (b8)",
            "value": 4375.055756325801,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.83 ms; ceiling 14803 tok/s"
          },
          {
            "name": "Qwen3-0.6B decode % of ceiling (b8)",
            "value": 29.5559423793717,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B prefill tok/s (b1)",
            "value": 18048.467184573074,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90752 tok/s"
          },
          {
            "name": "Qwen3-1.7B prefill % of ceiling (b1)",
            "value": 19.88774301169725,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B decode tok/s (b1)",
            "value": 394.75069165409764,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.53 ms; ceiling 708 tok/s"
          },
          {
            "name": "Qwen3-1.7B decode % of ceiling (b1)",
            "value": 55.73641665428472,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B prefill tok/s (b1)",
            "value": 20500.780000624232,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90752 tok/s"
          },
          {
            "name": "Qwen3-1.7B prefill % of ceiling (b1)",
            "value": 22.58996512126254,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B decode tok/s (b8)",
            "value": 2844.6874755777744,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.81 ms; ceiling 5486 tok/s"
          },
          {
            "name": "Qwen3-1.7B decode % of ceiling (b8)",
            "value": 51.8578213343818,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 9232.161822486445,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 23.738132402992843,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b1)",
            "value": 202.2450834868021,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 4.94 ms; ceiling 304 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b1)",
            "value": 66.60167481870015,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 11478.335392969668,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 29.51359070208422,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b8)",
            "value": 1511.0808176134838,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 5.30 ms; ceiling 2386 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b8)",
            "value": 63.3298080801635,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 6560.507406044052,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 31.704381996572735,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b1)",
            "value": 127.01812449403621,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.87 ms; ceiling 162 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b1)",
            "value": 78.5946032048366,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 7677.8415476003265,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 37.104023556168734,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b8)",
            "value": 974.8772495306813,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 8.20 ms; ceiling 1281 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b8)",
            "value": 76.13027269732801,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 43619.312450587495,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.7820400819197397,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 865.5550514448731,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.16 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 9.632254987598625,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 53555.401362953584,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 4.643554957163508,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 6767.21926210717,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.18 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 10.202681813064299,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 15704.1167728678,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 17.265103416153426,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 399.93178413549475,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.50 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 56.382372219044484,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 17285.59389907664,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 19.003779110507082,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 2998.431527495541,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 2.67 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 55.8235204799264,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 29742.402907319884,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.911697404434747,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 646.9470860158808,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.55 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 19.278258445411574,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 36448.83551665655,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 8.470173799345023,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5059.679212780323,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.49 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 19.895468213326275,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 14161.42564842181,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 27.814085028911556,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b1)",
            "value": 253.60636171558363,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.94 ms; ceiling 398 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b1)",
            "value": 63.78142775270701,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 15190.748928844516,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 29.835751911518525,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b8)",
            "value": 2009.5261903039827,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 3.97 ms; ceiling 3144 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b8)",
            "value": 63.923775493269524,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 25122.624906060028,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 16.596666379083644,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 588.1697034649619,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.70 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 49.75303341833568,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 29695.819686982857,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 19.617839061052646,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4535.413420383979,
            "unit": "tok/s",
            "extra": "ctx ~144; median step 1.77 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 48.47321909512268,
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
            "name": "Noah"
          },
          "distinct": true,
          "id": "868491478bcb00cc1ff81ecdddf03b9fd3bba887",
          "message": "Device-resident decode loop for the batched path\n\nBatchedModel's decode recipe takes only B int32 token ids and B positions\nper launch: the embedding rows, the RoPE rows and the per-slot mask rows\nare gathered on the device (gather_fwd_bf16 over the bf16 table, the full\n[head_dim, capacity] RoPE tables and the static mask patterns, with the\ngather indices computed by int32 nodes from the positions), the ScatterND\nquadruples come from the positions the same way, and IDS feeds the next\nlaunch through a per-slot ring, so BatchedGenerator::generate enqueues n\nlaunches back to back and reads the n x B ids once. Tied heads gather\nfrom the LM head's copy; scaled tables (Gemma, Granite) go through the\nsame cast / multiply nodes as batch 1; the sliding window and the\nper-layer masks are kept. RENG_DEVICE_LOOP=0 keeps the per-step path.\n\nBenches (medians of three, loop off to on): SmolLM2-1.7B batch 8 3050 to\n3202 tok/s (+5.0%), batch 64 13652 to 14509 (+6.3%); Qwen2.5-1.5B 3122 to\n3291 (+5.4%), 19573 to 21131 (+8.0%); Llama-3.2-3B 2030 to 2113 (+4.1%),\n12061 to 12869 (+6.7%); Qwen2.5-7B 1058 to 1084 (+2.5%), 7334 to 7682\n(+4.8%). Verified on module 5: reng-batch-test in all its forms (default,\nmin capacity, growth during a run, Gemma, OLMo-2, Qwen3, batch 8) with\nthe loop on and off, reng-cache-test, reng-generate --ref for\nSmolLM2-135M and Phi-3-mini (300 tokens) with identical verdicts,\nfree-running ids identical with and without the loop at batch 8 and 64\non Qwen2.5-1.5B, Phi-3-mini, Gemma-3-270m and OLMo-2, and identical again\nwith staggered per-slot prompts and positions.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_015f8r6hegh3UGeA9M8fCfUu",
          "timestamp": "2026-09-05T18:37:39-06:00",
          "tree_id": "d51d958f83eb25943be42ba650aea7b4eba13abc",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/868491478bcb00cc1ff81ecdddf03b9fd3bba887"
        },
        "date": 1788657139102,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 6579.783368857309,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 31.52692727247107,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b1)",
            "value": 132.2967844441753,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.56 ms; ceiling 163 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b1)",
            "value": 81.16158459068734,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 7298.321678231964,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 34.96979214388296,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b8)",
            "value": 1039.5013434093064,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.70 ms; ceiling 1293 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b8)",
            "value": 80.40390046387739,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 20987.453110258568,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 20.685014488001613,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b1)",
            "value": 427.5006977278965,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.34 ms; ceiling 792 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b1)",
            "value": 53.94755778872298,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 22548.732567451538,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 22.223795207233813,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b8)",
            "value": 3296.644810126201,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.43 ms; ceiling 6282 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b8)",
            "value": 52.480044942351356,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 26092.325655870354,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 23.39017459114767,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b1)",
            "value": 543.4667417375393,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.84 ms; ceiling 871 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b1)",
            "value": 62.41197687125332,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 27095.065763614235,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 24.2890697873449,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b8)",
            "value": 4172.004238169618,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.92 ms; ceiling 6790 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b8)",
            "value": 61.44616663095014,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B prefill tok/s (b1)",
            "value": 12583.216348743841,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 59826 tok/s"
          },
          {
            "name": "Gemma-2-2B prefill % of ceiling (b1)",
            "value": 21.032875915764034,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B decode tok/s (b1)",
            "value": 290.78312498109057,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.44 ms; ceiling 467 tok/s"
          },
          {
            "name": "Gemma-2-2B decode % of ceiling (b1)",
            "value": 62.25416627610186,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B prefill tok/s (b1)",
            "value": 13982.482461952492,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 59826 tok/s"
          },
          {
            "name": "Gemma-2-2B prefill % of ceiling (b1)",
            "value": 23.371752536540644,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B decode tok/s (b8)",
            "value": 2273.067647656366,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.52 ms; ceiling 3663 tok/s"
          },
          {
            "name": "Gemma-2-2B decode % of ceiling (b8)",
            "value": 62.05571315949299,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B prefill tok/s (b1)",
            "value": 22913.350123329317,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 156572 tok/s"
          },
          {
            "name": "Gemma-3-1B prefill % of ceiling (b1)",
            "value": 14.634377016825534,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B decode tok/s (b1)",
            "value": 503.06872314138684,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.99 ms; ceiling 1223 tok/s"
          },
          {
            "name": "Gemma-3-1B decode % of ceiling (b1)",
            "value": 41.14410589802855,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B prefill tok/s (b1)",
            "value": 25267.9800061186,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 156572 tok/s"
          },
          {
            "name": "Gemma-3-1B prefill % of ceiling (b1)",
            "value": 16.138240103382135,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B decode tok/s (b8)",
            "value": 3962.817625250578,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.02 ms; ceiling 9654 tok/s"
          },
          {
            "name": "Gemma-3-1B decode % of ceiling (b8)",
            "value": 41.04699047852823,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m prefill tok/s (b1)",
            "value": 38386.86401513402,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 582419 tok/s"
          },
          {
            "name": "Gemma-3-270m prefill % of ceiling (b1)",
            "value": 6.5909336763734965,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m decode tok/s (b1)",
            "value": 1047.6472421136514,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 0.95 ms; ceiling 4545 tok/s"
          },
          {
            "name": "Gemma-3-270m decode % of ceiling (b1)",
            "value": 23.049676099860942,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m prefill tok/s (b1)",
            "value": 46286.27036669348,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 582419 tok/s"
          },
          {
            "name": "Gemma-3-270m prefill % of ceiling (b1)",
            "value": 7.9472430463008115,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m decode tok/s (b8)",
            "value": 8111.342625869105,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 0.99 ms; ceiling 35168 tok/s"
          },
          {
            "name": "Gemma-3-270m decode % of ceiling (b8)",
            "value": 23.064277640803347,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 11264.939135445844,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 18.238056006579686,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b1)",
            "value": 280.6478059998186,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.56 ms; ceiling 482 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b1)",
            "value": 58.189615983711306,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 13303.53877639221,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 21.538570459390492,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b8)",
            "value": 2180.0315901905765,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.67 ms; ceiling 3798 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b8)",
            "value": 57.40494063795269,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 7087.864098179764,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 33.96138799920392,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b1)",
            "value": 132.4129173518087,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.55 ms; ceiling 163 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b1)",
            "value": 81.2328299414058,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 7354.663922419386,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 35.23975511003046,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b8)",
            "value": 1040.7118085007667,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.69 ms; ceiling 1293 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b8)",
            "value": 80.4975281588735,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 23748.78241105803,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 18.748269805403492,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b1)",
            "value": 607.6257141049998,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.65 ms; ceiling 989 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b1)",
            "value": 61.42566496954802,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 26062.636558416627,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 20.57492184567111,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b8)",
            "value": 4708.684452974194,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.70 ms; ceiling 7811 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b8)",
            "value": 60.28178840278444,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 12239.45031098722,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 25.133920033621646,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b1)",
            "value": 272.5635205132221,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.67 ms; ceiling 380 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b1)",
            "value": 71.68420055495741,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 12571.90478844701,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 25.816620975165794,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b8)",
            "value": 2102.0590390870652,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.81 ms; ceiling 2989 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b8)",
            "value": 70.32535055544584,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 7535.635756568116,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 34.22720455315415,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b1)",
            "value": 138.51840446207953,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.22 ms; ceiling 172 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b1)",
            "value": 80.55580604067731,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 7729.237620382403,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 35.1065531321865,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b8)",
            "value": 1092.680904037058,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.32 ms; ceiling 1363 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b8)",
            "value": 80.15636234189786,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 25584.362837433764,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 21.010037489576174,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b1)",
            "value": 528.2729887648552,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.89 ms; ceiling 950 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b1)",
            "value": 55.61953341098196,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 27233.403727072146,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 22.364240098931507,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b8)",
            "value": 4357.56281239555,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.84 ms; ceiling 7234 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b8)",
            "value": 60.23931257523527,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 8616.190643449712,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 20.592833496895107,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b1)",
            "value": 209.61931164548832,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 4.77 ms; ceiling 326 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b1)",
            "value": 64.23482713330277,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 9534.805321669088,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 22.788337275674266,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b8)",
            "value": 1619.6480414965977,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 4.94 ms; ceiling 2482 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b8)",
            "value": 65.26308451666713,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4230.775447263202,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 38.2021685342769,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b1)",
            "value": 72.96547382314832,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 13.71 ms; ceiling 87 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b1)",
            "value": 84.35210375923867,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4586.3721389250995,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 41.413060937911,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b8)",
            "value": 582.8644300936593,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 13.73 ms; ceiling 687 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b8)",
            "value": 84.83208183847054,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 30178.508233899473,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 9.522159418922449,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 802.2846459290962,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.25 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 32.4151981371102,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 37164.74649414675,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.726512064091423,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 6218.71286004313,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.29 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 31.794064483972317,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 20201.892664819035,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 19.91077432128147,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 427.53287963321475,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.34 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 53.95161891705829,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 22401.847802414828,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 22.07902711760955,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3303.8226260227652,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.42 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 52.59431024011184,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B prefill tok/s (b1)",
            "value": 2184.980407127058,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 4900 tok/s"
          },
          {
            "name": "Qwen2.5-32B prefill % of ceiling (b1)",
            "value": 44.59278389826931,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B decode tok/s (b1)",
            "value": 33.72597272382024,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 29.65 ms; ceiling 38 tok/s"
          },
          {
            "name": "Qwen2.5-32B decode % of ceiling (b1)",
            "value": 88.11470902517848,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B prefill tok/s (b1)",
            "value": 2341.9029891112077,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 4900 tok/s"
          },
          {
            "name": "Qwen2.5-32B prefill % of ceiling (b1)",
            "value": 47.79538231258577,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B decode tok/s (b8)",
            "value": 264.0495658001946,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 30.30 ms; ceiling 305 tok/s"
          },
          {
            "name": "Qwen2.5-32B decode % of ceiling (b8)",
            "value": 86.5845049337183,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 11677.262718249638,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 22.99747763864233,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 257.0493804471324,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.89 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 64.810985937602,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 13792.116970082017,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 27.16252166813841,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 2004.372883947922,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.99 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 63.5453839115598,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 7668.394651534368,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 34.595590005514225,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b1)",
            "value": 139.53501047875224,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.17 ms; ceiling 173 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b1)",
            "value": 80.58714499517428,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 8073.0699871624,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 36.421263126540936,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b8)",
            "value": 1082.7617898389058,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.39 ms; ceiling 1380 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b8)",
            "value": 78.48162358472408,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B prefill tok/s (b1)",
            "value": 23758.9344266403,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 259893 tok/s"
          },
          {
            "name": "Qwen3-0.6B prefill % of ceiling (b1)",
            "value": 9.14180180978001,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B decode tok/s (b1)",
            "value": 637.8836169779183,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.57 ms; ceiling 2024 tok/s"
          },
          {
            "name": "Qwen3-0.6B decode % of ceiling (b1)",
            "value": 31.511941406481117,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B prefill tok/s (b1)",
            "value": 23118.033477124944,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 259893 tok/s"
          },
          {
            "name": "Qwen3-0.6B prefill % of ceiling (b1)",
            "value": 8.895200284856413,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B decode tok/s (b8)",
            "value": 4698.416965675329,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.70 ms; ceiling 14803 tok/s"
          },
          {
            "name": "Qwen3-0.6B decode % of ceiling (b8)",
            "value": 31.74042774448733,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B prefill tok/s (b1)",
            "value": 16058.869809362397,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90752 tok/s"
          },
          {
            "name": "Qwen3-1.7B prefill % of ceiling (b1)",
            "value": 17.69539055925416,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B decode tok/s (b1)",
            "value": 393.264294516515,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.54 ms; ceiling 708 tok/s"
          },
          {
            "name": "Qwen3-1.7B decode % of ceiling (b1)",
            "value": 55.52654634392023,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B prefill tok/s (b1)",
            "value": 17382.596222133758,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90752 tok/s"
          },
          {
            "name": "Qwen3-1.7B prefill % of ceiling (b1)",
            "value": 19.15401474300173,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B decode tok/s (b8)",
            "value": 2987.2027998678445,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.68 ms; ceiling 5486 tok/s"
          },
          {
            "name": "Qwen3-1.7B decode % of ceiling (b8)",
            "value": 54.45583404681337,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 8268.777116986206,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 21.26103612436373,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b1)",
            "value": 201.73184265269305,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 4.96 ms; ceiling 304 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b1)",
            "value": 66.43265860061621,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 9178.085667695917,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 23.59908946300281,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b8)",
            "value": 1555.2928919676183,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 5.14 ms; ceiling 2386 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b8)",
            "value": 65.1827481420295,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 5791.723120884483,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 27.989146399520287,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b1)",
            "value": 126.79337264433737,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.89 ms; ceiling 162 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b1)",
            "value": 78.45553421356469,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 6398.841929589031,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 30.923115593845115,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b8)",
            "value": 988.1506455730986,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 8.10 ms; ceiling 1281 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b8)",
            "value": 77.16682089948918,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 34100.3060502468,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 2.9566886097496603,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 857.8407143936624,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.17 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 9.546406650842343,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 44721.69761817217,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.8776230853025195,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 7339.172428609848,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.09 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 11.064994078085304,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 13395.599147998033,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 14.727119516272444,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 398.6311752874084,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.51 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 56.199012418465735,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14958.597407114123,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.44547954716128,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3176.8726451198836,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.52 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 59.145661170090364,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 25241.111776319576,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 5.865663487010976,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 642.8747042424789,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.56 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 19.15690628228567,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 31271.07514720309,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.266938370860896,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5744.787071374648,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.39 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 22.589421930576535,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 13208.810028623904,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 25.94307765248395,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b1)",
            "value": 253.14644417372543,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.95 ms; ceiling 398 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b1)",
            "value": 63.66575952865383,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 13643.819407101964,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 26.797468173731446,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b8)",
            "value": 2073.825481445269,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.86 ms; ceiling 3144 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b8)",
            "value": 65.96915985856123,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 23517.229493986386,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 15.53609997090268,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 585.6635796674918,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.71 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 49.5410414365803,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 25042.281471184146,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 16.543589394120062,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4784.654208461291,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.67 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 51.13703432167147,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "bfe-noah",
            "email": "noah@it.bluefla.re"
          },
          "committer": {
            "name": "Noah",
            "email": "noah@it.bluefla.re"
          },
          "id": "fe19dfd717206760d0a918e4700f5d0a9d9eedf0",
          "message": "docs: review pass (readme)\n\nDocumentation-review pass over README.md and tools/oracle/README.md.\nNo number, name, command, path or claim changes.\n\nREADME.md:\n- Drop \"built from the ground up\" and \"fast silicon\" (marketing voice)\n  and the first-person \"we run\" / \"Our driver patches\".\n- Split the \"What works today\" internals into Benchmarks, How it works\n  and Multi-card sections and move them after Build, so the quick start\n  comes before the internals.\n- Move the SmolLM2-135M verification sentences from the reng-model\n  bullet into the verification paragraph under the benchmark table.\n- Turn the five-mechanism decode paragraph into a list and the two\n  sweep-table links into a list; split the four 45- to 95-word\n  sentences.\n\ntools/oracle/README.md:\n- Split one 45-word sentence.\n\ntools/profile/README.md: no change (dated measurement paragraphs).\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_015f8r6hegh3UGeA9M8fCfUu",
          "timestamp": "2026-09-06T02:02:10Z",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/fe19dfd717206760d0a918e4700f5d0a9d9eedf0"
        },
        "date": 1788705483791,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 6661.896471214258,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 31.92037089530786,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b1)",
            "value": 132.25086550959006,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.56 ms; ceiling 163 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b1)",
            "value": 81.1334141894995,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 6874.189081485154,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 32.937567558069105,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b8)",
            "value": 1040.0899248747924,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.69 ms; ceiling 1293 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b8)",
            "value": 80.44942637480176,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 21666.073670744066,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 21.35385582152373,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b1)",
            "value": 427.82667591574466,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.34 ms; ceiling 792 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b1)",
            "value": 53.988693925389576,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 22920.95982325075,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 22.59065850115452,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b8)",
            "value": 3300.504997892834,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.42 ms; ceiling 6282 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b8)",
            "value": 52.5414961568882,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 25845.631714690073,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 23.169028556451952,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b1)",
            "value": 544.4811122224576,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.84 ms; ceiling 871 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b1)",
            "value": 62.52846765602737,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 23847.515816620336,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 21.377839824347344,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b8)",
            "value": 4157.784545293963,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.92 ms; ceiling 6790 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b8)",
            "value": 61.2367359669339,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B prefill tok/s (b1)",
            "value": 14543.11442613325,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 59826 tok/s"
          },
          {
            "name": "Gemma-2-2B prefill % of ceiling (b1)",
            "value": 24.30885019187915,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B decode tok/s (b1)",
            "value": 291.36687727590254,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.43 ms; ceiling 467 tok/s"
          },
          {
            "name": "Gemma-2-2B decode % of ceiling (b1)",
            "value": 62.379142622056065,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B prefill tok/s (b1)",
            "value": 13918.122269154657,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 59826 tok/s"
          },
          {
            "name": "Gemma-2-2B prefill % of ceiling (b1)",
            "value": 23.26417432191616,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B decode tok/s (b8)",
            "value": 2274.8500487271103,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.52 ms; ceiling 3663 tok/s"
          },
          {
            "name": "Gemma-2-2B decode % of ceiling (b8)",
            "value": 62.104373466499396,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B prefill tok/s (b1)",
            "value": 23257.41543612832,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 156572 tok/s"
          },
          {
            "name": "Gemma-3-1B prefill % of ceiling (b1)",
            "value": 14.854125830456503,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B decode tok/s (b1)",
            "value": 502.7817065672789,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.99 ms; ceiling 1223 tok/s"
          },
          {
            "name": "Gemma-3-1B decode % of ceiling (b1)",
            "value": 41.120631887865784,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B prefill tok/s (b1)",
            "value": 24481.52832826019,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 156572 tok/s"
          },
          {
            "name": "Gemma-3-1B prefill % of ceiling (b1)",
            "value": 15.635946449361771,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B decode tok/s (b8)",
            "value": 3959.372629982956,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.02 ms; ceiling 9654 tok/s"
          },
          {
            "name": "Gemma-3-1B decode % of ceiling (b8)",
            "value": 41.01130710843125,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m prefill tok/s (b1)",
            "value": 39824.41167987769,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 582419 tok/s"
          },
          {
            "name": "Gemma-3-270m prefill % of ceiling (b1)",
            "value": 6.837757207235921,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m decode tok/s (b1)",
            "value": 1047.875396612651,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 0.95 ms; ceiling 4545 tok/s"
          },
          {
            "name": "Gemma-3-270m decode % of ceiling (b1)",
            "value": 23.054695811736533,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m prefill tok/s (b1)",
            "value": 42197.803357082514,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 582419 tok/s"
          },
          {
            "name": "Gemma-3-270m prefill % of ceiling (b1)",
            "value": 7.245262939570467,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m decode tok/s (b8)",
            "value": 8065.657222359679,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 0.99 ms; ceiling 35168 tok/s"
          },
          {
            "name": "Gemma-3-270m decode % of ceiling (b8)",
            "value": 22.934373027069853,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 11380.01432992742,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 18.4243639676515,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b1)",
            "value": 280.6997808695231,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.56 ms; ceiling 482 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b1)",
            "value": 58.20039247169465,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 12841.770587712786,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 20.790962861521404,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b8)",
            "value": 2183.6195438725845,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.66 ms; ceiling 3798 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b8)",
            "value": 57.49941920837991,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 6671.972970169504,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 31.968652279650513,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b1)",
            "value": 132.32422438734795,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.56 ms; ceiling 163 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b1)",
            "value": 81.17841847881492,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 7016.396338414088,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 33.61895136593673,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b8)",
            "value": 1038.7580553506207,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.70 ms; ceiling 1293 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b8)",
            "value": 80.34640822544455,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 23699.016564871985,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 18.708982590790093,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b1)",
            "value": 606.4362404781219,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.65 ms; ceiling 989 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b1)",
            "value": 61.3054195507011,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 25866.356869890937,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 20.41997055199608,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b8)",
            "value": 4705.7523392322455,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.70 ms; ceiling 7811 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b8)",
            "value": 60.24425072916662,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 12182.31172118267,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 25.016584964603695,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b1)",
            "value": 272.9584510877158,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.66 ms; ceiling 380 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b1)",
            "value": 71.78806728831185,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 13745.238580226925,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 28.226081935097078,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b8)",
            "value": 2102.941258254322,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.80 ms; ceiling 2989 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b8)",
            "value": 70.35486560285906,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 7212.216367617938,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 32.75821879804976,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b1)",
            "value": 138.46779024898268,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.22 ms; ceiling 172 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b1)",
            "value": 80.52637118868805,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 7480.4763584877455,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 33.976668027489396,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b8)",
            "value": 1092.63961166359,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.32 ms; ceiling 1363 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b8)",
            "value": 80.15333323574488,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 24358.27830360577,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 20.003169263668088,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b1)",
            "value": 527.8646380872614,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.89 ms; ceiling 950 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b1)",
            "value": 55.57653996888126,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 23777.02997264132,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 19.525844528167987,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b8)",
            "value": 4348.224797311153,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.84 ms; ceiling 7234 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b8)",
            "value": 60.11022307412673,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 8620.679785051432,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 20.603562617149485,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b1)",
            "value": 209.63624648960422,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 4.77 ms; ceiling 326 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b1)",
            "value": 64.24001657303413,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 9340.310557481635,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 22.323491677350027,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b8)",
            "value": 1618.0840409717334,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 4.94 ms; ceiling 2482 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b8)",
            "value": 65.20006372707385,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4133.790651148244,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 37.326435569332425,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b1)",
            "value": 72.95047357127845,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 13.71 ms; ceiling 87 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b1)",
            "value": 84.33476264246326,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4526.43481916842,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 40.87185150257719,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b8)",
            "value": 582.6910158796321,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 13.73 ms; ceiling 687 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b8)",
            "value": 84.80684254089677,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 33219.44427503105,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 10.48165938962926,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 805.0362058744247,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.24 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 32.52637109955755,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 35893.01470057304,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.325245282358377,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 6192.707926760488,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.29 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 31.661110519977274,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 19756.690186388936,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 19.471987410452375,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 427.16307287365316,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.34 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 53.90495192531268,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 21539.52990691158,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 21.229135610143636,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3292.694918374929,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.43 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 52.41716570950606,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B prefill tok/s (b1)",
            "value": 2175.9528884440124,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 4900 tok/s"
          },
          {
            "name": "Qwen2.5-32B prefill % of ceiling (b1)",
            "value": 44.40854325773197,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B decode tok/s (b1)",
            "value": 33.736655880226664,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 29.64 ms; ceiling 38 tok/s"
          },
          {
            "name": "Qwen2.5-32B decode % of ceiling (b1)",
            "value": 88.14262054683958,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B prefill tok/s (b1)",
            "value": 2343.7475252177624,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 4900 tok/s"
          },
          {
            "name": "Qwen2.5-32B prefill % of ceiling (b1)",
            "value": 47.83302704373478,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B decode tok/s (b8)",
            "value": 264.1626106690554,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 30.28 ms; ceiling 305 tok/s"
          },
          {
            "name": "Qwen2.5-32B decode % of ceiling (b8)",
            "value": 86.62157348172347,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 13318.167270354217,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 26.229113909460768,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 257.4792626503595,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.88 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 64.91937401999968,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 13994.97588565886,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 27.56203681884977,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 2001.4059094714853,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 4.00 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 63.45132080899497,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 7843.496748502934,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 35.38555461102892,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b1)",
            "value": 139.5904268126269,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.16 ms; ceiling 173 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b1)",
            "value": 80.61915018238669,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 8248.833805009814,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 37.21421305365982,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b8)",
            "value": 1082.487047989615,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.39 ms; ceiling 1380 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b8)",
            "value": 78.46170952181443,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B prefill tok/s (b1)",
            "value": 19794.927642580315,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 259893 tok/s"
          },
          {
            "name": "Qwen3-0.6B prefill % of ceiling (b1)",
            "value": 7.616558137577822,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B decode tok/s (b1)",
            "value": 634.9385996516488,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.57 ms; ceiling 2024 tok/s"
          },
          {
            "name": "Qwen3-0.6B decode % of ceiling (b1)",
            "value": 31.36645528494354,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B prefill tok/s (b1)",
            "value": 23224.877163874167,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 259893 tok/s"
          },
          {
            "name": "Qwen3-0.6B prefill % of ceiling (b1)",
            "value": 8.93631087472329,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B decode tok/s (b8)",
            "value": 4716.592312168252,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.70 ms; ceiling 14803 tok/s"
          },
          {
            "name": "Qwen3-0.6B decode % of ceiling (b8)",
            "value": 31.863212349664813,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B prefill tok/s (b1)",
            "value": 16766.50597018601,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90752 tok/s"
          },
          {
            "name": "Qwen3-1.7B prefill % of ceiling (b1)",
            "value": 18.4751402171239,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B decode tok/s (b1)",
            "value": 393.85787748032845,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.54 ms; ceiling 708 tok/s"
          },
          {
            "name": "Qwen3-1.7B decode % of ceiling (b1)",
            "value": 55.610356678112076,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B prefill tok/s (b1)",
            "value": 16764.500789756967,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90752 tok/s"
          },
          {
            "name": "Qwen3-1.7B prefill % of ceiling (b1)",
            "value": 18.472930693586136,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B decode tok/s (b8)",
            "value": 2983.732886441632,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.68 ms; ceiling 5486 tok/s"
          },
          {
            "name": "Qwen3-1.7B decode % of ceiling (b8)",
            "value": 54.39257853911802,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 8507.614813751363,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 21.875145904678455,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b1)",
            "value": 201.77355232291976,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 4.96 ms; ceiling 304 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b1)",
            "value": 66.44639408355276,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 8658.981358447716,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 22.264346088608708,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b8)",
            "value": 1551.7569246485994,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 5.16 ms; ceiling 2386 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b8)",
            "value": 65.03455478990631,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 5846.299329064073,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 28.25289199798667,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b1)",
            "value": 126.84245729791697,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.88 ms; ceiling 162 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b1)",
            "value": 78.48590616943244,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 6429.5533212871715,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 31.07153181133878,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b8)",
            "value": 987.9996291450767,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 8.10 ms; ceiling 1281 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b8)",
            "value": 77.15502770003498,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 41323.514694060716,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.582981484996606,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 862.2510324243574,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.16 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 9.595486495939593,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 34056.606070297355,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 2.9528995753412177,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 7326.011309358256,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.09 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 11.045151553877515,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 13597.692301645257,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 14.949300696397398,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 398.1896803918087,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.51 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 56.13677047990534,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14857.380392828269,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.334201578180643,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3181.865791163036,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.51 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 59.238621435430154,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 26518.357229204586,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.162476562573166,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 643.4462078065427,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.55 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 19.173936412954237,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30886.761951925117,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.177629631329328,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5754.864445291988,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.39 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 22.629047777199343,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 13103.747487509978,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 25.73672706854949,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b1)",
            "value": 253.01877891144773,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.95 ms; ceiling 398 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b1)",
            "value": 63.63365200324551,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 13378.544102409829,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 26.27644789908136,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b8)",
            "value": 2071.288473960902,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.86 ms; ceiling 3144 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b8)",
            "value": 65.88845670692383,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 23209.10057845057,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 15.33254190991108,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 587.3929305443562,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.70 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 49.68732651631458,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 26201.936430571102,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 17.30968794265002,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4802.1613627888655,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.67 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 51.32414584796334,
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
            "name": "Noah"
          },
          "distinct": true,
          "id": "5fd7fad062d328e65b1806c85d9accbd19ae169a",
          "message": "Tensor-parallel follow-up: teardown order, lifecycle, asserts\n\nReview follow-up on the reng-tp commit, from scratchpad/tp-review.md.\n\nTeardown order (review 1). Rust drops fields in declaration order, so\nRank::card released the device before Rank::comm destroyed the\ncommunicator, and every world-2 process printed\n\"hccl_device.cpp:45(operator->): The condition [ false ] failed. device\nnot initialized\" at exit. Comm now goes before Card in Rank and in the\nprobe's Worker, and Comm::drop calls hcclCommFinalize (exported by\nlibSynapse.so, already in the bindings) before hcclCommDestroy. The line\nis gone from a world-2 run.\n\nActivation (review 2). Recipe B open-codes SiLU as sigmoid + mult + mult,\nand nothing checked LayerWeights::act, so a gelu_pytorch_tanh model that\npassed the other asserts would have produced silently wrong logits with a\nVERDICT: PASS. TpModel::new now asserts the activation, and the # Panics\ndoc says so.\n\nLifecycle (review 3). A rank must never outlive its coordinator holding a\ncard. Each worker starts a watchdog thread that polls the hand-shake\ndirectory's abort file and its own parent id; on either it calls\nhcclCommAbort and leaves through _exit (abort_and_die, with a 10 s hard\ndeadline in case hcclCommAbort blocks, which on the peer of an aborted\nrank it does). Group::wait_all writes abort and waits out that grace\nbefore it kills anything, so a kill is never the first move at a rank\nstuck in a collective (Multi-Card risk 10), and Group has a Drop that\ndoes the same for a coordinator that returns early or panics. The\nworker's error path in bin/tp.rs leaves the same way instead of running\nthe destructor chain, which after an HCL failure can hang in libSynapse\nfor minutes while still holding the card.\n\nUniformity (review 5). The recipes and their per-layer bind lists come\nfrom layer 0, so every layer must agree with it on the presence of the\nattention biases and the q/k norms and on the norm epsilon; asserted.\n\nSecond prefill (review 6). The wide recipe's ScatterND is out of place,\nso its blocks alternate between the sequence's slot and a shared scratch\nbuffer and the parity is chosen to land the last block in the slot. A\nsecond prefill onto a non-empty sequence with an odd block count would\ntherefore read the scratch and write it over the keys already there.\nDocumented on TpModel::prefill and TpGenerator::prefill and rejected with\nan assert rather than fixed: seeding block 0 from the slot needs a\ndevice-side copy of the whole cache per layer.\n\nCoverage (review 4). reng-tp gains --prompt-file <json>, which takes the\nprompt from the \"prompt\" array of a generate.py reference file. With the\n1000-token prompt of ~/oracle/gen8_dsl8_1000.json the 8B distill prefills\nfour 256-row blocks, and world 2, world 1 and single-card reng-generate\nall give the oracle's eight ids exactly.\n\nCHANGELOG and README: the review's documentation notes (the id ring binds\nthe embedding and head recipes, not the layers; 18 GB rather than 24 GB\nis the measured strided-shard saving; the stale forward reference to the\ncoming multi-card path; the read-back tensor joining Bindings) plus the\nnew limitation, the lifecycle and the multi-block prefill result. The\ntwo-card 70B attainment is now 79% of 28.8 ms, the ceiling under\nreng-ceiling's convention of counting the embedding table as a lookup,\nrather than 81% of a ceiling that streamed it every token.\n\nChecks on this build, modules 5 and 6 only: reng-tp world 1 SmolLM2-135M\nand Llama-3.2-1B equal single-card reng-generate id for id; world 2 8B\n8/8 exact against gen8_dsl8c.json; world 2 8B over the 1000-token prompt\n8/8 exact against gen8_dsl8_1000.json and equal to both the single-card\nand world-1 ids; world 2 70B 8/8 exact against gen8_dsl70b.json and 27.4\ntok/s at batch 1 (36.48 ms/step; per layer A 90.4 us, all-reduces\n38.1 us, B 315.1 us); reng-generate --ref SmolLM2-135M 7/8 exact plus the\ndocumented near-tie; reng-cache-test and reng-batch-test PASS. SIGINT to\nthe coordinator mid-decode of a world-2 8B run: both workers gone in 11 s\nand both cards back to 768 MiB. cargo fmt --check, clippy with and\nwithout link-synapse, and cargo test all clean.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_015f8r6hegh3UGeA9M8fCfUu",
          "timestamp": "2026-09-06T09:19:09-06:00",
          "tree_id": "05d9a9994d2119354e38a2bfbc77f3517c71cfe0",
          "url": "https://github.com/blueflare-energy/reciprocating-engine/commit/5fd7fad062d328e65b1806c85d9accbd19ae169a"
        },
        "date": 1788709377100,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 6643.458850415881,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 31.83202732874926,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b1)",
            "value": 132.41446960440746,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.55 ms; ceiling 163 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b1)",
            "value": 81.23378221912843,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill tok/s (b1)",
            "value": 7268.863778703077,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B prefill % of ceiling (b1)",
            "value": 34.82864508721197,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode tok/s (b8)",
            "value": 1039.712564638392,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.69 ms; ceiling 1293 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Llama-8B decode % of ceiling (b8)",
            "value": 80.42023811537437,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 20674.749524601903,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 20.376817101411785,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b1)",
            "value": 427.2184905502941,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.34 ms; ceiling 792 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b1)",
            "value": 53.91194524328618,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill tok/s (b1)",
            "value": 22318.282376270243,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B prefill % of ceiling (b1)",
            "value": 21.996665906774876,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode tok/s (b8)",
            "value": 3293.815481872017,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.43 ms; ceiling 6282 tok/s"
          },
          {
            "name": "DeepSeek-R1-Distill-Qwen-1.5B decode % of ceiling (b8)",
            "value": 52.43500421685974,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 23632.822741274504,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 21.18538060924784,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b1)",
            "value": 542.5502033147545,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.84 ms; ceiling 871 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b1)",
            "value": 62.3067211666235,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base prefill tok/s (b1)",
            "value": 25012.960206407726,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 111553 tok/s"
          },
          {
            "name": "Falcon3-1B-Base prefill % of ceiling (b1)",
            "value": 22.422589461191908,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Falcon3-1B-Base decode tok/s (b8)",
            "value": 4159.35131276845,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.92 ms; ceiling 6790 tok/s"
          },
          {
            "name": "Falcon3-1B-Base decode % of ceiling (b8)",
            "value": 61.25981165185974,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B prefill tok/s (b1)",
            "value": 13499.328672057329,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 59826 tok/s"
          },
          {
            "name": "Gemma-2-2B prefill % of ceiling (b1)",
            "value": 22.564159832938234,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B decode tok/s (b1)",
            "value": 291.0349476493157,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.44 ms; ceiling 467 tok/s"
          },
          {
            "name": "Gemma-2-2B decode % of ceiling (b1)",
            "value": 62.30807934365281,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B prefill tok/s (b1)",
            "value": 13531.144750834872,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 59826 tok/s"
          },
          {
            "name": "Gemma-2-2B prefill % of ceiling (b1)",
            "value": 22.61734048393459,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-2-2B decode tok/s (b8)",
            "value": 2269.0781744053547,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.53 ms; ceiling 3663 tok/s"
          },
          {
            "name": "Gemma-2-2B decode % of ceiling (b8)",
            "value": 61.94679884364432,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B prefill tok/s (b1)",
            "value": 24561.189406298487,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 156572 tok/s"
          },
          {
            "name": "Gemma-3-1B prefill % of ceiling (b1)",
            "value": 15.686824659806968,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B decode tok/s (b1)",
            "value": 502.83471499218354,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.99 ms; ceiling 1223 tok/s"
          },
          {
            "name": "Gemma-3-1B decode % of ceiling (b1)",
            "value": 41.124967248319415,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B prefill tok/s (b1)",
            "value": 25535.09616931165,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 156572 tok/s"
          },
          {
            "name": "Gemma-3-1B prefill % of ceiling (b1)",
            "value": 16.308842770317117,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-1B decode tok/s (b8)",
            "value": 3965.1821690652373,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.02 ms; ceiling 9654 tok/s"
          },
          {
            "name": "Gemma-3-1B decode % of ceiling (b8)",
            "value": 41.07148249825378,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m prefill tok/s (b1)",
            "value": 39610.713336339206,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 582419 tok/s"
          },
          {
            "name": "Gemma-3-270m prefill % of ceiling (b1)",
            "value": 6.801065707548482,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m decode tok/s (b1)",
            "value": 1048.375261600342,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 0.95 ms; ceiling 4545 tok/s"
          },
          {
            "name": "Gemma-3-270m decode % of ceiling (b1)",
            "value": 23.06569352699486,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m prefill tok/s (b1)",
            "value": 52832.46232862276,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 582419 tok/s"
          },
          {
            "name": "Gemma-3-270m prefill % of ceiling (b1)",
            "value": 9.071208709056567,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Gemma-3-270m decode tok/s (b8)",
            "value": 8178.214278606267,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 0.98 ms; ceiling 35168 tok/s"
          },
          {
            "name": "Gemma-3-270m decode % of ceiling (b8)",
            "value": 23.254424505036447,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 11615.722497485605,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 18.805977992321008,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b1)",
            "value": 280.7035100140582,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.56 ms; ceiling 482 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b1)",
            "value": 58.20116567384983,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct prefill tok/s (b1)",
            "value": 12937.276633781263,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 61766 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct prefill % of ceiling (b1)",
            "value": 20.945588163639815,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "granite-3.1-2b-instruct decode tok/s (b8)",
            "value": 2178.7081625692795,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.67 ms; ceiling 3798 tok/s"
          },
          {
            "name": "granite-3.1-2b-instruct decode % of ceiling (b8)",
            "value": 57.37009193008028,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 6465.586587059805,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 30.97975521031465,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b1)",
            "value": 132.31839664969158,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.56 ms; ceiling 163 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b1)",
            "value": 81.17484327156588,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B prefill tok/s (b1)",
            "value": 7195.060377447528,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20870 tok/s"
          },
          {
            "name": "Llama-3.1-8B prefill % of ceiling (b1)",
            "value": 34.475017264925654,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.1-8B decode tok/s (b8)",
            "value": 1039.7417018428005,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.69 ms; ceiling 1293 tok/s"
          },
          {
            "name": "Llama-3.1-8B decode % of ceiling (b8)",
            "value": 80.42249183529297,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 23782.856573767363,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 18.77516935691778,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b1)",
            "value": 606.1479636918506,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.65 ms; ceiling 989 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b1)",
            "value": 61.276277279594154,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B prefill tok/s (b1)",
            "value": 29943.40199914901,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 126672 tok/s"
          },
          {
            "name": "Llama-3.2-1B prefill % of ceiling (b1)",
            "value": 23.638558383957744,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-1B decode tok/s (b8)",
            "value": 4734.813193031694,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.69 ms; ceiling 7811 tok/s"
          },
          {
            "name": "Llama-3.2-1B decode % of ceiling (b8)",
            "value": 60.61629524755351,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 12722.160731778684,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 26.12517411834777,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b1)",
            "value": 273.2333287965216,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.66 ms; ceiling 380 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b1)",
            "value": 71.8603601203424,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B prefill tok/s (b1)",
            "value": 13354.601753842635,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 48697 tok/s"
          },
          {
            "name": "Llama-3.2-3B prefill % of ceiling (b1)",
            "value": 27.423902547374347,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Llama-3.2-3B decode tok/s (b8)",
            "value": 2104.8642549608917,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.80 ms; ceiling 2989 tok/s"
          },
          {
            "name": "Llama-3.2-3B decode % of ceiling (b8)",
            "value": 70.41920034084302,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 7052.821111914716,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 32.03423820795803,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b1)",
            "value": 138.4402730039588,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.22 ms; ceiling 172 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b1)",
            "value": 80.51036844983521,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 prefill tok/s (b1)",
            "value": 7553.294560098817,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22017 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 prefill % of ceiling (b1)",
            "value": 34.307411651816324,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Mistral-7B-v0.3 decode tok/s (b8)",
            "value": 1092.9966788570096,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.32 ms; ceiling 1363 tok/s"
          },
          {
            "name": "Mistral-7B-v0.3 decode % of ceiling (b8)",
            "value": 80.17952679987728,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 25709.956341280387,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 21.11317565412737,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b1)",
            "value": 528.8381036843953,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.89 ms; ceiling 950 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b1)",
            "value": 55.67903186881887,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B prefill tok/s (b1)",
            "value": 27273.43162653233,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 121772 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B prefill % of ceiling (b1)",
            "value": 22.397111258304566,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "OLMo-2-0425-1B decode tok/s (b8)",
            "value": 4352.516352072395,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.84 ms; ceiling 7234 tok/s"
          },
          {
            "name": "OLMo-2-0425-1B decode % of ceiling (b8)",
            "value": 60.16954988588508,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 8568.776076191414,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 20.4795119225147,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b1)",
            "value": 209.79612314334855,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 4.77 ms; ceiling 326 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b1)",
            "value": 64.28900847714498,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill tok/s (b1)",
            "value": 9404.542743141641,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 41841 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct prefill % of ceiling (b1)",
            "value": 22.477007628792414,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode tok/s (b8)",
            "value": 1617.3922127025423,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 4.95 ms; ceiling 2482 tok/s"
          },
          {
            "name": "Phi-3-mini-4k-instruct decode % of ceiling (b8)",
            "value": 65.17218677748578,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4161.507859950435,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 37.576710606417166,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b1)",
            "value": 72.96509990003541,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 13.71 ms; ceiling 87 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b1)",
            "value": 84.35167148353943,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 prefill tok/s (b1)",
            "value": 4623.76548904432,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 11075 tok/s"
          },
          {
            "name": "phi-4 prefill % of ceiling (b1)",
            "value": 41.75070756584531,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "phi-4 decode tok/s (b8)",
            "value": 583.0915729758658,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 13.72 ms; ceiling 687 tok/s"
          },
          {
            "name": "phi-4 decode % of ceiling (b8)",
            "value": 84.8651409900974,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 31235.350803492795,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 9.855622668675364,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b1)",
            "value": 804.3214684396509,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.24 ms; ceiling 2475 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b1)",
            "value": 32.49749312503596,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B prefill tok/s (b1)",
            "value": 36192.75668422605,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 316929 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B prefill % of ceiling (b1)",
            "value": 11.419822222038977,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-0.5B decode tok/s (b8)",
            "value": 6207.06848350794,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.29 ms; ceiling 19559 tok/s"
          },
          {
            "name": "Qwen2.5-0.5B decode % of ceiling (b8)",
            "value": 31.734530933096504,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 22580.105128030078,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 22.254715675128782,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b1)",
            "value": 428.5198550590512,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.33 ms; ceiling 792 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b1)",
            "value": 54.07616821979475,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B prefill tok/s (b1)",
            "value": 23337.962713000787,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 101462 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B prefill % of ceiling (b1)",
            "value": 23.001652192037472,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-1.5B decode tok/s (b8)",
            "value": 3301.027790476133,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.42 ms; ceiling 6282 tok/s"
          },
          {
            "name": "Qwen2.5-1.5B decode % of ceiling (b8)",
            "value": 52.54981861194394,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B prefill tok/s (b1)",
            "value": 2193.9277906375846,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 4900 tok/s"
          },
          {
            "name": "Qwen2.5-32B prefill % of ceiling (b1)",
            "value": 44.775389077720085,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B decode tok/s (b1)",
            "value": 33.736218956183016,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 29.64 ms; ceiling 38 tok/s"
          },
          {
            "name": "Qwen2.5-32B decode % of ceiling (b1)",
            "value": 88.1414790101584,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B prefill tok/s (b1)",
            "value": 2351.801352599964,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 4900 tok/s"
          },
          {
            "name": "Qwen2.5-32B prefill % of ceiling (b1)",
            "value": 47.99739583296374,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-32B decode tok/s (b8)",
            "value": 264.1431377796113,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 30.29 ms; ceiling 305 tok/s"
          },
          {
            "name": "Qwen2.5-32B decode % of ceiling (b8)",
            "value": 86.61518812567475,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 13125.333709046012,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 25.84934292878098,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b1)",
            "value": 257.4495149144285,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.88 ms; ceiling 397 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b1)",
            "value": 64.91187359307102,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B prefill tok/s (b1)",
            "value": 14077.258911337933,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50776 tok/s"
          },
          {
            "name": "Qwen2.5-3B prefill % of ceiling (b1)",
            "value": 27.724086957546824,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-3B decode tok/s (b8)",
            "value": 2003.3070530227953,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.99 ms; ceiling 3154 tok/s"
          },
          {
            "name": "Qwen2.5-3B decode % of ceiling (b8)",
            "value": 63.511593474728215,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 7642.829557019211,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 34.48025432334226,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b1)",
            "value": 139.56995787449372,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.16 ms; ceiling 173 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b1)",
            "value": 80.60732853791498,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B prefill tok/s (b1)",
            "value": 8203.249936963552,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 22166 tok/s"
          },
          {
            "name": "Qwen2.5-7B prefill % of ceiling (b1)",
            "value": 37.0085636470427,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen2.5-7B decode tok/s (b8)",
            "value": 1081.5220982066087,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 7.40 ms; ceiling 1380 tok/s"
          },
          {
            "name": "Qwen2.5-7B decode % of ceiling (b8)",
            "value": 78.39176724424354,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B prefill tok/s (b1)",
            "value": 23312.867792362485,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 259893 tok/s"
          },
          {
            "name": "Qwen3-0.6B prefill % of ceiling (b1)",
            "value": 8.970167312571624,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B decode tok/s (b1)",
            "value": 638.1937966805108,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.57 ms; ceiling 2024 tok/s"
          },
          {
            "name": "Qwen3-0.6B decode % of ceiling (b1)",
            "value": 31.527264522412327,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B prefill tok/s (b1)",
            "value": 23722.765934580722,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 259893 tok/s"
          },
          {
            "name": "Qwen3-0.6B prefill % of ceiling (b1)",
            "value": 9.127885142465312,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-0.6B decode tok/s (b8)",
            "value": 4716.590747979807,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.70 ms; ceiling 14803 tok/s"
          },
          {
            "name": "Qwen3-0.6B decode % of ceiling (b8)",
            "value": 31.863201782699242,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B prefill tok/s (b1)",
            "value": 16126.84773672921,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90752 tok/s"
          },
          {
            "name": "Qwen3-1.7B prefill % of ceiling (b1)",
            "value": 17.77029595349698,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B decode tok/s (b1)",
            "value": 393.1091817817505,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.54 ms; ceiling 708 tok/s"
          },
          {
            "name": "Qwen3-1.7B decode % of ceiling (b1)",
            "value": 55.50464536136085,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B prefill tok/s (b1)",
            "value": 16870.706382572098,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90752 tok/s"
          },
          {
            "name": "Qwen3-1.7B prefill % of ceiling (b1)",
            "value": 18.58995944260465,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-1.7B decode tok/s (b8)",
            "value": 2980.466893165851,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.68 ms; ceiling 5486 tok/s"
          },
          {
            "name": "Qwen3-1.7B decode % of ceiling (b8)",
            "value": 54.33304043617041,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 8371.014890073533,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 21.523914293182536,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b1)",
            "value": 201.87569521731118,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 4.95 ms; ceiling 304 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b1)",
            "value": 66.48003093504015,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B prefill tok/s (b1)",
            "value": 9177.97494310664,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 38892 tok/s"
          },
          {
            "name": "Qwen3-4B prefill % of ceiling (b1)",
            "value": 23.5988047631664,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-4B decode tok/s (b8)",
            "value": 1552.6046565464135,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 5.15 ms; ceiling 2386 tok/s"
          },
          {
            "name": "Qwen3-4B decode % of ceiling (b8)",
            "value": 65.07008346432679,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 5820.732357269276,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 28.12933676207136,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b1)",
            "value": 126.88322863510855,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 7.88 ms; ceiling 162 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b1)",
            "value": 78.5111341208092,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B prefill tok/s (b1)",
            "value": 6402.895869729481,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 20693 tok/s"
          },
          {
            "name": "Qwen3-8B prefill % of ceiling (b1)",
            "value": 30.942706710636738,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "Qwen3-8B decode tok/s (b8)",
            "value": 988.0630710917709,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 8.10 ms; ceiling 1281 tok/s"
          },
          {
            "name": "Qwen3-8B decode % of ceiling (b8)",
            "value": 77.15998201885266,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 12844.798804791471,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 1.1137164066707292,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b1)",
            "value": 856.6824820961393,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.17 ms; ceiling 8986 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b1)",
            "value": 9.533517362280058,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M prefill tok/s (b1)",
            "value": 42546.93781087778,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 1153328 tok/s"
          },
          {
            "name": "SmolLM2-135M prefill % of ceiling (b1)",
            "value": 3.68905916034261,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-135M decode tok/s (b8)",
            "value": 7329.8381649040075,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.09 ms; ceiling 66328 tok/s"
          },
          {
            "name": "SmolLM2-135M decode % of ceiling (b8)",
            "value": 11.050921159969116,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 13698.833918869297,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 15.060495773860593,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b1)",
            "value": 398.60955768433683,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 2.51 ms; ceiling 709 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b1)",
            "value": 56.19596476936353,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B prefill tok/s (b1)",
            "value": 14721.858588641007,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 90959 tok/s"
          },
          {
            "name": "SmolLM2-1.7B prefill % of ceiling (b1)",
            "value": 16.185208928783172,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-1.7B decode tok/s (b8)",
            "value": 3170.5876866338267,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 2.52 ms; ceiling 5371 tok/s"
          },
          {
            "name": "SmolLM2-1.7B decode % of ceiling (b8)",
            "value": 59.02865049115886,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 27908.57499688209,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 6.48554274409177,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b1)",
            "value": 644.631016328685,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.55 ms; ceiling 3356 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b1)",
            "value": 19.20924230642204,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M prefill tok/s (b1)",
            "value": 30820.4586686212,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 430320 tok/s"
          },
          {
            "name": "SmolLM2-360M prefill % of ceiling (b1)",
            "value": 7.162221722541824,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM2-360M decode tok/s (b8)",
            "value": 5732.186513884788,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.40 ms; ceiling 25431 tok/s"
          },
          {
            "name": "SmolLM2-360M decode % of ceiling (b8)",
            "value": 22.53987452243721,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 12685.47552694276,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 24.91521007123202,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b1)",
            "value": 253.0302607507675,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 3.95 ms; ceiling 398 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b1)",
            "value": 63.63653965992764,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B prefill tok/s (b1)",
            "value": 13372.026371150803,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 50915 tok/s"
          },
          {
            "name": "SmolLM3-3B prefill % of ceiling (b1)",
            "value": 26.26364659390667,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "SmolLM3-3B decode tok/s (b8)",
            "value": 2072.2555523031756,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 3.86 ms; ceiling 3144 tok/s"
          },
          {
            "name": "SmolLM3-3B decode % of ceiling (b8)",
            "value": 65.91921982866575,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 21942.279120532883,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 14.495646355503764,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b1)",
            "value": 585.2007378430353,
            "unit": "tok/s",
            "extra": "ctx ~160; mean step 1.71 ms; ceiling 1182 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b1)",
            "value": 49.50188983692474,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B prefill tok/s (b1)",
            "value": 24422.296421377534,
            "unit": "tok/s",
            "extra": "128 tokens; ceiling 151372 tok/s"
          },
          {
            "name": "TinyLlama-1.1B prefill % of ceiling (b1)",
            "value": 16.134010973467934,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          },
          {
            "name": "TinyLlama-1.1B decode tok/s (b8)",
            "value": 4783.034814664127,
            "unit": "tok/s",
            "extra": "ctx ~144; mean step 1.67 ms; ceiling 9357 tok/s"
          },
          {
            "name": "TinyLlama-1.1B decode % of ceiling (b8)",
            "value": 51.119726697634725,
            "unit": "%",
            "extra": "HbmBandwidth bound"
          }
        ]
      }
    ]
  }
}