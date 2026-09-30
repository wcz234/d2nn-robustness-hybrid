# Editorial revision of the Chinese Markdown source

This revision changes interpretations, not measured results. The shipped PDF/DOCX and
English manuscript remain historical snapshots and have not been regenerated to include
these edits. Rebuild publication formats only after reviewing this working Markdown and
the supplementary evidence. Existing figures/tables retain their historical cohort labels.

Corrections:

- Sections 3.5/4.1 consistently distinguish the latent continuous path from the four-level
  QAT deployment path. The 61.74% latent result does not establish a cost at the same
  deployment assumption. Continuous-path checkpoint selection and quantization-plus-error
  performance remain questions to test.
- Sections 3.7/4.2 and conclusion 3 distinguish identical optical **architecture** from
  identical trained phase **weights**. Retrained-head ablations show that electronic
  nonlinearity is not necessary in the tested recipe; they do not completely identify the
  causal contribution of learned readout or prove dimension equivalence.
- The tested phase-filtering recipe is specifically training-time smoothing followed by
  unfiltered validation/test. Its failure does not establish that deploying smoothed masks
  or other phase-filtered methods fails.
- The limitations disclose subpixel interpolation/sampling sensitivity and the retrospective
  v2 declared-vs-observed seed discrepancy.

The new v5 design, scripts and pipeline pilots are not inserted as completed formal results.
No headline accuracy, confidence interval, or historical per-seed record has been replaced.
