# Dynamics Encoder

`DynamicsEncoder` uses a Transformer encoder to learn a latent physical context
from a fixed window of past transitions. Learned positional embeddings preserve
the order of the window, and the last transition is held out for the prediction
loss:

```text
past:  (s[t-H], a[t-H], ..., s[t])  ->  z[t]
query: (s[t], a[t])                 ->  predict s[t+1]
```

For the current SRB locomotion task, the deployable state should start with
the sensor-side contract

```python
sensor_state = torch.cat((obs["proprio"], obs["proprio_dyn"]), dim=-1)
```

Keep `state`/`state_dyn` out of this input when the encoder will run on a real
robot. In simulation, `info["metrics/gravity_magnitude"] / 9.80665` can be
passed as `physics_target=... .unsqueeze(-1)` to teach the latent a known
gravity label. Friction, mass, actuator gain, and delay can be appended to the
same target vector after those parameters are randomized and logged per
environment.

The runtime sequence is:

```python
runtime.reset(sensor_state_after_reset)
latent = runtime.begin_step()
policy_obs = torch.cat((base_policy_obs, latent), dim=-1)
action = policy(policy_obs)
next_sensor_state = torch.cat((next_obs["proprio"], next_obs["proprio_dyn"]), dim=-1)
runtime.observe(action, next_sensor_state, done, physics_target=gravity_target)

batch = runtime.drain_batch()
if batch is not None:
    metrics = trainer.update(batch)
```

The encoder update is an auxiliary update. It should be run after collecting a
rollout batch, under `torch.enable_grad()` if the surrounding RL rollout uses
`torch.no_grad()`.

The optional FPO integration is enabled with a fresh run, for example:

```bash
conda activate srb
python -m srb agent train --algo fpo --env locomotion_velocity_tracking_c \
  'env.domain=moon' 'env.num_envs=256' 'env.sim.device=cuda' \
  'env.physics_conditioning.gravity_magnitude_range=[1.62496,3.72076]' \
  'env.physics_conditioning.gravity_interval_s=[2.56,2.56]' \
  'agent.dynamics_encoder.enabled=true' \
  'agent.dynamics_encoder.update_interval=64' --headless
```

This changes the actor observation dimension by `latent_dim`; do not resume a
checkpoint trained with the old observation contract. The current Isaac Sim
GPU path still needs a normal smoke run after enabling it.
