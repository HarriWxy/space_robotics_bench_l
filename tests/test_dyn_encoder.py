"""CPU tests for the standalone dynamics encoder."""

import torch

from dyn.dyn_encoder import (
    DynamicsEncoder,
    DynamicsEncoderConfig,
    DynamicsEncoderRuntime,
    DynamicsEncoderTrainer,
    DynamicsTransitionBatch,
)


def _make_config() -> DynamicsEncoderConfig:
    return DynamicsEncoderConfig(
        state_dim=5,
        action_dim=2,
        latent_dim=3,
        hidden_dim=16,
        history_length=4,
        physics_dim=1,
    )


def test_encoder_predicts_shapes_and_both_losses() -> None:
    torch.manual_seed(7)
    encoder = DynamicsEncoder(_make_config())
    batch_size = 6
    history = _make_config().history_length
    states = torch.randn(batch_size, history + 1, 5)
    actions = torch.randn(batch_size, history, 2)
    query_state = torch.randn(batch_size, 5)
    query_action = torch.randn(batch_size, 2)
    target_next_state = torch.randn(batch_size, 5)
    physics_target = torch.randn(batch_size, 1)

    output = encoder(states, actions, query_state, query_action)
    assert isinstance(encoder.temporal_encoder, torch.nn.TransformerEncoder)
    assert output.latent.shape == (batch_size, 3)
    assert output.predicted_delta.shape == (batch_size, 5)
    assert output.predicted_next_state.shape == (batch_size, 5)
    assert output.physics is not None
    assert output.physics.shape == (batch_size, 1)

    loss, metrics = encoder.loss(
        batch=DynamicsTransitionBatch(
            context_states=states,
            context_actions=actions,
            query_state=query_state,
            query_action=query_action,
            target_next_state=target_next_state,
            physics_target=physics_target,
        )
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert {
        "transition_loss",
        "physics_loss",
        "transition_rmse",
        "physics_rmse",
    } <= metrics.keys()


def test_runtime_masks_done_transition_and_resets_history() -> None:
    torch.manual_seed(11)
    encoder = DynamicsEncoder(_make_config())
    runtime = DynamicsEncoderRuntime(encoder, num_envs=2, device="cpu")
    initial_state = torch.zeros(2, 5)
    runtime.reset(initial_state)

    action = torch.ones(2, 2)
    next_state = torch.ones(2, 5)
    latent = runtime.begin_step()
    assert torch.isfinite(latent).all()
    runtime.observe(
        action,
        next_state,
        done=torch.tensor([False, True]),
        physics_target=torch.ones(2, 1),
    )
    batch = runtime.drain_batch()
    assert batch is not None
    assert batch.context_states.shape == (2, 5, 5)
    assert batch.sample_mask is not None
    assert torch.equal(batch.sample_mask, torch.tensor([1.0, 0.0]))
    assert torch.equal(runtime.history.mask[0, -1], torch.ones(1))
    assert torch.equal(runtime.history.mask[1], torch.zeros(4, 1))
    assert torch.equal(runtime.history.current_state[1], next_state[1])


def test_trainer_updates_parameters() -> None:
    torch.manual_seed(13)
    encoder = DynamicsEncoder(_make_config())
    runtime = DynamicsEncoderRuntime(encoder, num_envs=3, device="cpu")
    runtime.reset(torch.zeros(3, 5))
    for _ in range(5):
        runtime.begin_step()
        runtime.observe(
            torch.randn(3, 2),
            torch.randn(3, 5),
            physics_target=torch.randn(3, 1),
        )
    batch = runtime.drain_batch()
    assert batch is not None

    before = [parameter.detach().clone() for parameter in encoder.parameters()]
    metrics = DynamicsEncoderTrainer(encoder, learning_rate=1.0e-3).update(batch)
    assert metrics["valid_samples"] == 15.0
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, encoder.parameters())
    )
