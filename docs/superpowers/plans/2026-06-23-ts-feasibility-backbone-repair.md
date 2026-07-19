# TS Feasibility and Backbone Repair Implementation Plan

Goal: modify TS so hard 50-50 cases such as `R_50_50_1` do not accept drone physical-location illegal solutions and can repair truck backbone ordering around remaining late nodes.

Key implementation points:

- Add a TS-level `_non_delay_feasible()` gate.
- Penalize non-time-window hard violations in TS/SearchEvaluator search costs.
- Filter generated candidates before acceptance.
- Add a violation-directed truck backbone rechain neighborhood.
- Keep shared initial solution construction unchanged.

Source plan: `/Users/minz/.codex/attachments/5f340d2f-bd75-4752-8acc-19d0c56ce703/pasted-text.txt`.
