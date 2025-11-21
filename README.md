# cnn-copy-traj
Scripts for imitating a trajectory in Montezuma's Revenge

To run:
```[bash]
python main.py
```

# Things We Learned Along the Way
Because of floating-point precision limitations, reproducing a trajectory with zero error is infeasible.

We aim to mitigate this by conditioning the agent on an RNN (GRU), so that small deviations from the reference trajectory do not derail the behavior and the agent can still progress and accumulate the desired score.
