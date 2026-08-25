# IEEE Paper Figure Plan

This note records where visual evidence is most useful in
`PAPER_Faithfulness_Decoupling.md`. A figure is recommended only when it makes
an important result easier to understand than the corresponding table.

| Paper location | Figure | Status | Purpose |
|---|---|---|---|
| Section III, before the environment details | Telemetry degradation data flow | Present | Separates the true SUMO state from the stale policy observation |
| Section III-B | Model conditions and training process | Present | Explains why B0, B1, B2, B3 and H5' are needed |
| Section III-C | Observation graph and GAT design | Present | Shows the node types, attention flow and action outputs |
| Section III-F | DEF occlusion protocol | Present | Explains comprehensiveness, sufficiency and random controls |
| Section IV-B, after Table IV | Clean faithfulness audit | Present | Shows why the uniform baseline gives a misleading result |
| Section IV-B, after Table V | Capability spectrum | **Added** | Shows the uniform score changing sign while corrected scores stay near zero |
| Section IV-C, after Table VI | AoI manipulation check | **Added** | Shows that the nominal 5-60 s ladder collapsed at the 60 s encoding cap |
| Section IV-C, after the H2-H4 discussion | Stale-attention drift | Present | Compares pickups, aggregate DEF and WAMSN under clean and degraded telemetry |
| Section IV-D | Coupled versus decoupled explanation | Present | Shows the small clean-data gain and its loss under degradation |

## Graphs to Add Only After More Experiments

1. **Realised AoI dose-response plot.** Add this only after rerunning the sweep
   with an AoI encoding cap above the geographic tunnel floor. Plot realised
   AoI on the x-axis, rather than nominal outage settings.
2. **Multi-city robustness plot.** Add a forest plot of clean-versus-degraded
   WAMSN and faithfulness effects after evaluating more maps. The current
   single-district experiment cannot support this graph.
3. **Training stability plot.** Add episode pickups and policy entropy over
   training epochs if training behaviour becomes part of the main argument.
   At present it is better placed in an appendix because the paper studies
   explanation reliability rather than dispatch optimisation.

The two added figures are generated from committed audit artifacts by
`scripts/plot_ieee_paper_figures.py`. Both PNG and vector PDF versions are
saved under `docs/figures/`.
