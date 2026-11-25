# Estimating SEED-Bench category scores with 200 questions

This task asks how accurately 200 questions can estimate nine category scores on the full 14,232-question SEED-Bench image benchmark. The agent chooses questions and fits an estimator using cached responses from 37 training model families. The evaluator measures mean absolute error on 16 reserved families. Each family and each category receive equal weight. GPU requirement: 0. [Task proposal](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/194)

## Results

| Candidate | Round | Mean absolute error (percentage points) | Reward |
| --- | ---: | ---: | ---: |
| Stratified sample with identity calibration | 1 | 8.45 | 0.5133 |
| Reference questions with fitted calibration | 2 | 3.04 | 0.6975 |
| Selected questions with identity calibration | 3 | 4.89 | 0.6158 |
| Round 3 questions with fitted calibration | 4 | 2.85 | 0.7079 |
| Best retained candidate | 39 | 1.77 | 0.7792 |

All 50 rounds used the same feedback panel. Round 39 first reached the best reported reward. The error fell by 6.68 percentage points from the reference. A separate refit from public training data reproduced the question list and estimator byte for byte. A separate evaluation reproduced both endpoint rewards and all nine category errors.

## Scoring and review

Version 2.0.0 uses `1 - ln(1 + E) / ln(101)`, where `E` is the mean absolute error in percentage points. The evaluator rounds the error to two decimal places before applying the formula. It then rounds the reward to four decimal places. Both steps use half-up rounding. [Scoring amendment](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/194#discussioncomment-19028056)

The assigned private task repository contains the runnable task, both original trajectories, all 50 scores, and the reproduction instructions at commit `d192a307`. Maintainers can review the evidence there. [Submission handoff](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/194#discussioncomment-19893274) · [Private reproduction record](https://github.com/OpenRSI-Foundation/seed-bench-profile-200-d194/blob/d192a307e5bf4aec1f5ab19110cb1df5e32c451f/validation/20261125-reproduction/README.md)
