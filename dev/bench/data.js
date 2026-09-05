window.BENCHMARK_DATA = {
  "lastUpdate": 1788605162611,
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
      }
    ]
  }
}