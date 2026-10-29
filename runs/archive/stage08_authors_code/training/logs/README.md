# logs — launcher logs of the training

`gpu0.log`: the ε = 5 queue (self-attention, then SlaterNet) on GPU 0.
`gpu1.log`: the ε = 10 queue on GPU 1.
Lines starting with `===` give the start and end time of each run; the rest
is what the authors' code printed. Written by `launch_fig5.sh`, appended to,
never truncated.
