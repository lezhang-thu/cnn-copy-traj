# cnn-copy-traj
Scripts for imitating a trajectory in Montezuma's Revenge

To run:
```[bash]
python main.py
```

# Things We Learned Along the Way

1.	A CNN alone cannot precisely replicate every action in a trajectory. This is because identical states can correspond to different actions. For example, when the agent chooses **NOOP**, the states $s_t$ and $s_{t+1}$ remain identical, yet the action $a_{t+1}$ may differ from $a_t$.
2.	Due in part to floating-point precision limits, using an RNN (e.g., a GRU) to reproduce a trajectory with **zero error** is practically infeasible.
3.	A key advantage of RNNs is that **small deviations from the reference trajectory do not derail the agent’s behavior**. Even with slight inconsistencies, the policy can still progress through the environment and accumulate the desired score.
4.	For RNNs, **trajectory truncation heavily influences learning**. Truncating sequences every 64 steps results in worse performance, though the degradation is not severe. So far, only truncating at 1024 steps reliably reproduces the same score as the original Go-Explore trajectory.
