# Profile

Scripts over a SynapseAI profiler trace (`HABANA_PROFILE=1` after
`hl-prof-config --gaudi2 --invoc json --merged json -o DIR -s NAME`; the
merged trace is `DIR/NAME.json`).

- `timeline.py TRACE [l1_]`: per node of one layer, first start, last end,
  launch count and busy time over the whole trace.
- `stepgaps.py TRACE`: splits the trace into launches at the first layer's
  norm, prints each launch's wall time and engine busy fractions, and for
  the last launch (a decode step) the per-layer wall versus busy time and
  layer 5's node sequence with the idle gap before every node.

Measured on Qwen2.5-1.5B at batch 1, context 132 (2026-09-05): the device
window of a decode step is 2.32 ms of a 2.59 ms step (the rest is host
uploads, launch and readback); engines are busy 75% of the window (MME
72%, TPC 3%); each layer spends 56 of 74 us busy and the other 18 us in
1 to 3 us gaps between consecutive nodes (norm to projections 1.7,
projections to scatter 2.6, softmax to av 1.6, up to down 2.7). The
projections stream weights at 1.2 to 2.2 TB/s (q/k/v 5.5 us for 11 MB,
gate plus up 25 us for 55 MB, down 15 us for 27 MB, o 3.8 us for 4.7 MB).

Fused q/k/v projection (2026-09-06, same model, batch 1, the decode
step's device window): the three projection gemms of a Qwen2 layer ran
concurrently in a 5.8 us window (q 5.7, k 4.6, v 3.7 us); one gemm over
their concatenated weight runs in 3.9 us, and the launch's device window
goes from 2340 to 2294 us. The three bias adds cost nothing in either
form: the TPC fuser merges them with the RoPE kernels into two fused TPC
nodes of 0.1 to 0.2 us (`fusedTPCNode_*`, which carry no layer prefix,
so `stepgaps.py` does not list them under the layer). A bias given to
the gemm as its third input is instead split out by the graph compiler
into an `add_fwd_bf16` stage of its own (`<node>_add_bias_complex`),
which is not fused and costs a 1.9 us handoff.
