# gpu_locks — one lock per GPU, shared by every stage

`gpu0.lock` and `gpu1.lock` are empty files. Every job that uses a GPU holds
the lock of its GPU for as long as it runs: the training queues of
`learn/archive/stage08_authors_code/launch_fig5.sh`, every run of its
`analysis/evaluate_weights.py`, and the jobs of our own Stage 8
(`learn/stage08_own_code/`). A second job for the same GPU stops at once, so
two jobs never share a GPU (two processes on one GPU crashed the machine on
2026-10-01). The operating system releases a lock when its job ends, for any
reason; nothing needs to be cleaned up, and the files must not be deleted
while a job runs.

Until 2026-10-28 this folder was `runs/stage08/gpu_locks/`.
