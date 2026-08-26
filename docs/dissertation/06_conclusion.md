# 6. Conclusion

This dissertation examined whether graph-attention explanations remain
trustworthy when fleet telemetry becomes stale. Tunnel entry triggered signal
loss, but the degradation was applied at the observation boundary: SUMO kept
the true state while the policy received frozen last-known vehicle data. This
separation made it possible to compare clean and degraded versions of the same
decision.

The strongest result is also the narrowest. Longer outages increase WAMSN for
all six trained GAT policies. Attention-based explanations therefore carry
more stale-data exposure when observation outages last longer. This does not
mean that AoI causes lower faithfulness. H1 and H2 are unsupported in every
training run, and DEF remains close to the type-matched random baseline.

The exact attention response is not stable across learned policies. Seeds 42
and 43 shift attention toward stale nodes, while seed 44 shifts it away, for
both clean-trained and degradation-trained GATs. The WAMSN-DEF relationship is
also mixed. Degradation-aware training does not remove this variation and does
not provide a consistent mitigation.

The construct-validity audit supports this conclusion by showing that request
and taxi occlusions are not equivalent. Deleting a request can delete an
action; deleting a taxi only removes information. Type-matched controls are
therefore required before DEF can be interpreted.

The central lesson is about trust rather than performance. A complete and
plausible attention map can outlive the freshness of its data, yet its response
to that degradation may depend on the particular trained checkpoint. Stable
dispatch output is not enough to validate the explanation. Graph-attention
weights should be treated as model internals unless each released policy passes
explicit, freshness-aware, action-aware, and cross-seed faithfulness checks.
