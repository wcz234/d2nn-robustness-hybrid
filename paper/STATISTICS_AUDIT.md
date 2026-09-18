# Statistics audit

## Statistics review scope

- Inputs reviewed: formal clean summary, formal robustness summary, Methods statistical wording, Results draft, and Figure 2 legend/source data.
- Independent unit: independently trained model identified by training seed.
- Replication: three training seeds per method; deployment draws, test samples, and timing repeats are nested measurements rather than independent replicates.
- Analyses: seed-level means and paired seed-level differences with two-sided 95% Student-t intervals.
- Hypothesis tests: none.

## Major statistical issues

- P0: none found. The analysis does not treat test samples or deployment draws as independent replicates.
- P1: three training seeds provide limited precision and do not support a reliable normality assessment. The manuscript now describes t intervals as marginal uncertainty summaries.
- P1: 14 condition-specific intervals are not simultaneous confidence bands. The Methods, Results, and Figure 2 legend now state that no simultaneous-coverage adjustment was applied.
- P1: four-level phase quantization produces wide intervals, including an unbounded accuracy interval above 100%. The figure retains the raw interval and the Results interprets it as seed instability.
- P2: CPU wall-clock intervals show large seed-to-seed variability. The manuscript restricts them to fixed-environment software simulation overhead.

## Ready-to-paste statistical analysis

We treated the independently trained model, indexed by training seed, as the independent unit. We first averaged deployment draws within each seed and then summarized seed-level estimates across seeds 42, 43, and 44. Cross-method contrasts paired models with the same training seed. We report means, standard deviations across training seeds, individual seed values, and two-sided 95% Student-t confidence intervals. These intervals provide marginal descriptive uncertainty for three training seeds; they do not provide simultaneous coverage across conditions, and three observations do not support a reliable normality assessment. We performed no hypothesis tests and report no p values or multiple-comparison correction. Test samples, deployment draws, and CPU timing repeats were not treated as independent replicates.

## Reviewer-risk note

- A reviewer may request more training seeds because the four-level phase-quantization result remains imprecise.
- A reviewer may request another dataset before accepting cross-dataset or broad edge-intelligence claims.
- No statistical wording can substitute for missing physical hardware, energy, or real-device measurements; those claims remain outside scope.
