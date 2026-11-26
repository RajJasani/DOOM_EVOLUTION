"""Example Forward-Pass Script for DOOM Autonomous Agent Neural Network.

Demonstrates:
  1. Instantiating DoomNetwork with custom configuration.
  2. Single-step decision inference with hidden state propagation (evaluation mode).
  3. Processing Gymnasium observation dictionary matching docs/CONTRACT.md.
  4. Batched multi-step sequential forward pass for PPO training (Member 4 pipeline).
  5. Inspecting tensor shapes and intermediate representations.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch

# Ensure repository root and env are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "env"))

from doom_env import spaces as S
from network import DoomNetwork, NetworkConfig


def main() -> None:
    print("=" * 75)
    print("  DOOM Autonomous Agent: Neural Network Forward-Pass Demonstration")
    print("=" * 75)

    # 1. Initialize Network Configuration & Model
    config = NetworkConfig(
        vitals_dim=17,
        num_rays=64,
        max_entities=32,
        action_dim=9,
        rnn_hidden_dim=256,
        fusion_hidden_dim=256,
    )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n[1] Initializing DoomNetwork on device: '{device}'")
    model = DoomNetwork(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"    - Total Parameters:     {total_params:,}")
    print(f"    - Trainable Parameters: {trainable_params:,}")
    print(f"    - Model Memory Footprint: {model_size_mb:.2f} MB (Budget constraint: <= 2048 MB VRAM)")

    # 2. Single-Step Inference (Agent Evaluation Mode)
    print("\n[2] Executing Single-Step Inference (Evaluation Agent act() interface)...")
    obs_space = S.observation_space()
    sample_obs = obs_space.sample()

    # Reset memory between maps
    hx = model.init_hidden(batch_size=1, device=device)
    print(f"    - Initialized recurrent hidden state shape: {tuple(hx.shape)}")

    # Execute step 1
    step_res = model.step(sample_obs, hx=hx, deterministic=True)
    print("    - Step 1 Output:")
    print(f"        Action (MultiBinary 9 buttons): {step_res.action.cpu().numpy().astype(int)}")
    print(f"        Action shape:                  {tuple(step_res.action.shape)}")
    print(f"        State Value V(s):              {step_res.value.item():.4f}")
    print(f"        Action Log Probability:        {step_res.log_prob.item():.4f}")
    print(f"        Next Hidden State hx shape:    {tuple(step_res.hx.shape)}")

    # Check convenience method for submission/agent.py
    action_np, next_hx = model.get_action(sample_obs, hx=hx, deterministic=True)
    print(f"    - Submission format action: {action_np} (type: {type(action_np)}, dtype: {action_np.dtype})")
    assert action_np.shape == (9,)
    assert action_np.dtype == "int8"
    print("    - Action format matches official CONTRACT.md MultiBinary(9) specification!")

    # 3. Batched Sequential Forward Pass (Member 4 PPO Rollouts & Update)
    print("\n[3] Executing Batched Sequential Evaluation (Member 4 PPO Training Pipeline)...")
    batch_size = 4
    seq_len = 8
    print(f"    - Simulating rollout batch: Batch Size = {batch_size}, Sequence Length = {seq_len} steps")

    # Generate synthetic batched sequence observation
    batched_obs = {
        "vitals": torch.randn(batch_size, seq_len, 17, device=device),
        "rays": {
            "distance": torch.rand(batch_size, seq_len, 64, device=device),
            "hit_type": torch.randint(0, 6, (batch_size, seq_len, 64), device=device),
        },
        "visible_entities": {
            "features": torch.randn(batch_size, seq_len, 32, 5, device=device),
            "entity_type": torch.randint(0, 12, (batch_size, seq_len, 32), device=device),
            "mask": (torch.rand(batch_size, seq_len, 32, device=device) > 0.5).float(),
        },
    }
    simulated_actions = torch.randint(0, 2, (batch_size, seq_len, 9), device=device).float()
    simulated_dones = torch.zeros(batch_size, seq_len, device=device)
    simulated_dones[:, 4] = 1.0  # Episode terminated at step 4

    batch_hx = model.init_hidden(batch_size=batch_size, device=device)
    eval_res = model.evaluate_actions(
        obs=batched_obs,
        hx=batch_hx,
        actions=simulated_actions,
        dones=simulated_dones,
    )

    print("    - Evaluation Result Shapes:")
    print(f"        Values V(s_t):      {tuple(eval_res.values.shape)}  (expected: [B, T, 1])")
    print(f"        Log Probs log_pi:   {tuple(eval_res.log_probs.shape)}  (expected: [B, T])")
    print(f"        Entropy H(pi):      {tuple(eval_res.entropy.shape)}  (expected: [B, T])")
    print(f"        Final Hidden hx:    {tuple(eval_res.hx.shape)}  (expected: [1, B, 256])")

    # 4. Check intermediate representations
    print("\n[4] Inspecting Internal Component Representations...")
    vitals_t = batched_obs["vitals"][:, 0]  # (B, 17)
    vitals_emb = model.vitals_encoder(vitals_t)
    print(f"    - VitalsEncoder output:        {tuple(vitals_emb.shape)}")

    ray_dist_t = batched_obs["rays"]["distance"][:, 0]
    ray_hit_t = batched_obs["rays"]["hit_type"][:, 0]
    ray_emb = model.ray_encoder(ray_dist_t, ray_hit_t)
    print(f"    - RaycastEncoder output:       {tuple(ray_emb.shape)}")

    ent_feats_t = batched_obs["visible_entities"]["features"][:, 0]
    ent_type_t = batched_obs["visible_entities"]["entity_type"][:, 0]
    ent_mask_t = batched_obs["visible_entities"]["mask"][:, 0]
    enemy_emb = model.enemy_attention(ent_feats_t, ent_type_t, ent_mask_t, context_query=vitals_emb)
    print(f"    - EnemyAttentionModule output: {tuple(enemy_emb.shape)}")

    fused_emb = model.fusion(vitals_emb, ray_emb, enemy_emb)
    print(f"    - FeatureFusion output:        {tuple(fused_emb.shape)}")

    print("\n" + "=" * 75)
    print("  Demonstration completed successfully! All components verified.")
    print("=" * 75)


if __name__ == "__main__":
    main()
