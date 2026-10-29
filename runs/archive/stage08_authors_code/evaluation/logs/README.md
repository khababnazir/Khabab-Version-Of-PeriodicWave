# logs — launcher logs of the evaluation

`gpu0.log` (ε = 5 runs) and `gpu1.log` (ε = 10 runs), written by
`analysis/launch_evaluation.sh`: start, end and exit code of each
evaluation, the sanity check, and progress every 1000 steps. Appended to,
never truncated; starts that found an evaluation already finished say so.
