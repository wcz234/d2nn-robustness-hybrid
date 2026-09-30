# Historical protocol alignment

This is a retrospective disclosure. It does not amend frozen files or establish that a
corrected design was registered before the historical runs.

`FORMAL_EXPERIMENT_PROTOCOL_V2.json` declares seeds 42–44 and retains the v1 experiment stage
and output directory. Stored v2 seed metrics contain seeds 42–46 for baseline, robust, hybrid,
and hybrid QAT. Its statistics section and the run records must be read alongside the
training section rather than treating the entire file as an internally consistent execution
specification. The discrepancy prevents an unqualified claim that all five-seed settings
were fixed by this JSON before execution. It does not by itself invalidate the recorded
per-seed measurements.

Run the audit from the repository root:

```bash
python simulator/audit_protocol_alignment.py --output artifacts/protocol_alignment.json
```

The audit records SHA-256 values for each protocol and its seed-metric file. v1, v3, and v4
match their declared seed lists in the checked robustness summaries; v2 does not. The audit
is limited to those summary files and does not independently authenticate training history.

The v2/v3/v4 builders previously referenced protocol files in the repository root although
the files live under `docs/`. They now resolve those paths and require an explicit, unused
output path. Generated files are labeled `regenerated_design_not_original_preregistration`.
The derived v2 design corrects seed count/stage metadata; it must never be substituted for the
historical file or used to claim retroactive preregistration.

Fresh supplementary work uses prospective v5 designs with matching training/statistical
seed counts, explicit nominal phase paths, and a new output tree. Historical protocol,
result, and manifest hashes are retained.
