"""Recurrent memory architecture using Gated Recurrent Units (GRU).

Maintains hidden states across consecutive game tics/decision steps,
supporting both single-step online inference and multi-step sequential PPO training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import NetworkConfig


class RecurrentMemory(nn.Module):
    """GRU-based recurrent memory module for temporal tracking and navigation persistence."""

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.fusion_hidden_dim
        self.hidden_dim = config.rnn_hidden_dim
        self.num_layers = config.rnn_num_layers

        dropout = config.rnn_dropout if self.num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if config.orthogonal_init:
            self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization to recurrent weights."""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def init_hidden(self, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
        """Create a zero-initialized hidden state tensor.

        Returns:
            Tensor of shape (num_layers, batch_size, hidden_dim)
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through recurrent memory.

        Args:
            x: Input tensor.
               Can be (batch_size, input_dim) for single-step,
               or (batch_size, seq_len, input_dim) for sequential rollout.
            hx: Initial hidden state of shape (num_layers, batch_size, hidden_dim).
                If None, initialized to zeros on x.device.
            dones: Optional episode termination mask of shape (batch_size, seq_len)
                   or (batch_size,). 1.0 indicates episode boundary (resets hidden state).

        Returns:
            output: (batch_size, hidden_dim) if single-step,
                    or (batch_size, seq_len, hidden_dim) if sequential.
            next_hx: Updated hidden state of shape (num_layers, batch_size, hidden_dim).
        """
        is_single_step = (x.ndim == 2)
        if is_single_step:
            # (B, D) -> (B, 1, D)
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x

        batch_size, seq_len, _ = x_seq.shape

        if hx is None:
            hx = self.init_hidden(batch_size=batch_size, device=x.device)
        else:
            # Validate and align hidden state device and shape
            if hx.device != x.device:
                hx = hx.to(x.device)
            if hx.shape[1] != batch_size:
                raise ValueError(
                    f"Hidden state batch dimension {hx.shape[1]} does not match input batch size {batch_size}."
                )

        # Sequential processing with episode boundary resets (for PPO updates)
        if dones is not None and seq_len > 1:
            dones_flat = dones.reshape(batch_size, seq_len)
            outputs = []
            curr_hx = hx
            for t in range(seq_len):
                # Reset hidden state at episode boundaries where done == 1
                done_t = dones_flat[:, t].view(1, batch_size, 1)
                curr_hx = curr_hx * (1.0 - done_t.float())
                step_x = x_seq[:, t : t + 1, :]
                step_out, curr_hx = self.gru(step_x, curr_hx)
                outputs.append(step_out)
            output = torch.cat(outputs, dim=1)
            next_hx = curr_hx
        else:
            if dones is not None and is_single_step:
                # Reset if done flag is passed on single step
                done_mask = dones.reshape(1, batch_size, 1)
                hx = hx * (1.0 - done_mask.float())
            output, next_hx = self.gru(x_seq, hx)

        if is_single_step:
            output = output.squeeze(1)

        return output, next_hx
