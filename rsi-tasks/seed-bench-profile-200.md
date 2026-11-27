# Estimating SEED-Bench category scores with 200 questions

This task asks how accurately 200 questions can estimate nine category scores on the full 14,232-question SEED-Bench image benchmark. The agent chooses questions and fits an estimator using cached responses from 37 training model families. The evaluator measures mean absolute error on 16 reserved families. Each family and each category receive equal weight. The task uses 0 GPUs. [Task proposal](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/194)

## Results

| Run and candidate | Round | Mean absolute error (percentage points) | Reward |
| --- | ---: | ---: | ---: |
| Stratified reference, both runs | 1 | 8.45 | 0.5133 |
| Capped rerun, best candidate | 46 | 2.03 | 0.7598 |
| Original run, reference questions with fitted calibration | 2 | 3.04 | 0.6975 |
| Original run, selected questions with identity calibration | 3 | 4.89 | 0.6158 |
| Original run, round 3 questions with fitted calibration | 4 | 2.85 | 0.7079 |
| Original run, best candidate | 39 | 1.77 | 0.7792 |

The capped rerun reduced the error by 6.42 percentage points from the reference. It completed 50 scored submissions after two baseline validation trials. The runner enforced the 50-submission limit and stopped the agent at the 24-hour limit. Both runs used the same 16 families for feedback throughout the search.

An independent evaluation reproduced the baseline and round 46 rewards and all nine category errors for both candidates. A fresh fit on public training data reproduced all 200 selected questions and the full score report. Each fitted parameter differed from its archived value by less than `1e-10`. [Capped rerun and reproduction record](https://github.com/OpenRSI-Foundation/seed-bench-profile-200-d194/blob/8d5d4cc3bb092ec6c58d69e0a83f6963369fb051/validation/20261125-capped-rerun/round-46/README.md)

The original run also scored 50 submissions. Its runner reported an unlimited allowance. Its best result reduced error by 6.68 percentage points. A separate refit reproduced its round 39 question list and estimator byte for byte. A separate evaluation reproduced both endpoint rewards and all nine category errors. [Original reproduction record](https://github.com/OpenRSI-Foundation/seed-bench-profile-200-d194/blob/d192a307e5bf4aec1f5ab19110cb1df5e32c451f/validation/20261125-reproduction/README.md)

## Scoring and review

Version 2.0.0 uses `1 - ln(1 + E) / ln(101)`, where `E` is the mean absolute error in percentage points. The evaluator rounds the error to two decimal places before applying the formula. It then rounds the reward to four decimal places. Both steps use half-up rounding. The accepted proposal used a linear reward. The logarithmic amendment awaits maintainer review. [Scoring amendment](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/194#discussioncomment-19028056)
