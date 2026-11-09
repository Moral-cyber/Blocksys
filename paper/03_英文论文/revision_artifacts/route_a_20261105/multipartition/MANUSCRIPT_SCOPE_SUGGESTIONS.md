# Minimal scope edits for root integration

No manuscript source was edited by this review. `combined_development.json` covers every old/new configuration, and `AGGREGATION_VERIFICATION.json` independently checks aggregation arithmetic, exact CSV/JSON agreement and 35 unchanged input/source hashes. It does not claim raw-logit reevaluation.

## One results sentence

Place after the primary paired-analysis paragraph, with the compact table in the appendix:

> Across three label-skew partitions per task, including two new realizations, the same TRAIN-fit/development-holdout protocol lowers client-macro NLL in all nine configurations per task; partition-mean reductions are 0.0391--0.0463 on CIFAR-10 and 0.0103--0.0138 on FashionMNIST (Table~\ref{tab:partitiondev}).

This refers to nine model-seed/partition configurations, not nine independent model seeds. The same three model-seed numbers are reused across partitions. Avoid a claim that all clients improve: the counts are 177/180 and 176/180 client-seed cells, respectively. The complete JSON/CSV retains one adverse CIFAR macro-accuracy run, five adverse CIFAR weighted-accuracy runs and one adverse Fashion weighted-accuracy run. These do not alter the reported NLL pattern.

Use `table_development_compact.tex` without substituting its development values into the protected TEST tables. The table contains all six partition summaries, each with three-seed sample SD. Appendix metadata can state: original CIFAR/Fashion partitions 20261217/20261219; additional partitions 20261105/20261106 for both tasks; original three model-seed numbers and the same training/fitting settings.

## Three-sentence limitations paragraph

Replace only the current paragraph beginning “The construction assumes ...”:

> The primary TEST evaluation uses one fixed partition per task, while the additional partition checks concern label skew rather than feature or concept shift. The local prior must remain useful at serving time: the frozen-center reallocation diagnostic reverses the NLL gain as client label distributions drift (Appendix~\ref{app:sensitivity}). The primary TEST benefit is most consistent in NLL; weighted CIFAR accuracy, Fashion's lower client-accuracy decile, and one CIFAR seed's accuracy favor the zero-center control.

The wording keeps a precise TEST/development distinction, retains the adverse drift observation and leaves the limitations paragraph at three sentences. It does not imply that re-estimating a prior has been tested or that the method resolves every form of data heterogeneity.
