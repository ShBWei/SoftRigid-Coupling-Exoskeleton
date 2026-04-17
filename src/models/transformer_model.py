"""Transformer-based sequence-to-sequence motion predictor.

Architecture
------------
* Positional encoding is added to the input embedding.
* A standard PyTorch ``nn.Transformer`` (encoder + decoder) processes the
  IMU history and autoregressively predicts future frames.
* Linear projection layers map raw feature dimensions to/from d_model.

Input  : (batch, seq_len, input_dim)
Output : (batch, predict_frames, output_dim)
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """Transformer encoder-decoder for multi-step motion prediction."""

    def __init__(
        self,
        input_dim: int = 30,
        output_dim: int = 15,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        predict_frames: int = 1,
    ) -> None:
        super().__init__()
        self.predict_frames = predict_frames
        self.d_model = d_model
        self.output_dim = output_dim

        # Input / output projection
        self.src_embed = nn.Linear(input_dim, d_model)
        self.tgt_embed = nn.Linear(output_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        src : (batch, src_len, input_dim)  – IMU history
        tgt : (batch, tgt_len, output_dim) – teacher-forced target (training only).
              If None the model runs in autoregressive inference mode.

        Returns
        -------
        preds : (batch, predict_frames, output_dim)
        """
        # Encode source
        src_emb = self.pos_enc(self.src_embed(src))  # (B, S, d_model)

        if tgt is not None:
            # Teacher-forced training
            tgt_emb = self.pos_enc(self.tgt_embed(tgt))
            tgt_mask = self._make_causal_mask(tgt.size(1), tgt.device)
            out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
            return self.fc_out(out)  # (B, tgt_len, output_dim)

        # Autoregressive inference: start token = zeros
        batch = src.size(0)
        device = src.device
        # Encode memory once
        memory = self.transformer.encoder(src_emb)  # (B, S, d_model)

        pred_token = torch.zeros(batch, 1, self.output_dim, device=device)
        preds = []
        for _ in range(self.predict_frames):
            tgt_emb = self.pos_enc(self.tgt_embed(pred_token))
            tgt_mask = self._make_causal_mask(pred_token.size(1), device)
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            step_pred = self.fc_out(out[:, -1:, :])  # (B, 1, output_dim)
            preds.append(step_pred)
            pred_token = torch.cat([pred_token, step_pred], dim=1)

        return torch.cat(preds, dim=1)  # (B, predict_frames, output_dim)

    @classmethod
    def from_config(cls, cfg: dict) -> "TransformerPredictor":
        """Build from a config dict (e.g. loaded from YAML)."""
        data_cfg = cfg["data"]
        m_cfg = cfg["model"]["transformer"]
        return cls(
            input_dim=data_cfg["input_dim"],
            output_dim=data_cfg["output_dim"],
            d_model=m_cfg["d_model"],
            nhead=m_cfg["nhead"],
            num_encoder_layers=m_cfg["num_encoder_layers"],
            num_decoder_layers=m_cfg["num_decoder_layers"],
            dim_feedforward=m_cfg["dim_feedforward"],
            dropout=m_cfg["dropout"],
            max_seq_len=m_cfg.get("max_seq_len", 512),
            predict_frames=data_cfg.get("predict_frames", 1),
        )
