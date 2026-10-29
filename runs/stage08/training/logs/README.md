# logs — what the two GPU queues of launch_fig5.sh printed

`gpu0.log` and `gpu1.log`, appended to, never truncated: start and end of
every run with its exit code, and a progress line every 1000 iterations
(energy, accepted moves, step width, iterations per second).

`gpu0.log` also holds the end of `attention_eps5` (see `../README.md`): a
failed start on 2026-10-29 00:17, which found no working GPU, and the restart
on GPU 0 at 05:41.

`gpu_monitor_2026-10-29.csv`: `nvidia-smi` every 10 seconds during that
restart (05:41–06:25), for both GPUs: temperature, power, clock, fan, PCIe
lanes, and why the clock is lowered (`0x20` = software thermal slowdown,
`0x01` = idle).  GPU 0 averaged 86.8 °C (maximum 91 °C; the card's own
maximum operating temperature is 93 °C), its fan reached 100 %, and it was in
thermal slowdown in 238 of its 246 samples during training.  That is why this
part ran at about 10 iterations per second instead of 12.
