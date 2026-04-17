"""LSTM-based sequence-to-sequence motion predictor.

Architecture
------------
Encoder  : bidirectional or unidirectional multi-layer LSTM that encodes the
           IMU history window into a context vector.
Decoder  : single LSTM step that takes the last encoder hidden state and the
           last observed frame as seed input to produce the next-frame prediction.

Input  : (batch, seq_len, input_dim)   – normalised IMU channels
Output : (batch, predict_frames, output_dim) – predicted joint angles / positions
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Encodes a variable-length IMU history into a fixed context vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Project bidirectional output back to hidden_dim for the decoder
        if bidirectional:
            self.proj_h = nn.Linear(hidden_dim * 2, hidden_dim)
            self.proj_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        outputs : (batch, seq_len, hidden_dim * num_directions)
        (h_n, c_n) : final hidden & cell state, each (num_layers, batch, hidden_dim)
        """
        outputs, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            # Merge forward + backward for each layer
            h_n = self.proj_h(
                torch.cat([h_n[0::2], h_n[1::2]], dim=-1)
            )  # (num_layers, batch, hidden_dim)
            c_n = self.proj_c(torch.cat([c_n[0::2], c_n[1::2]], dim=-1))

        return outputs, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """Decodes a context vector to predict one or more future frames."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x      : (batch, 1, input_dim)  – seed input (last observed frame)
        hidden : encoder final state

        Returns
        -------
        pred   : (batch, 1, output_dim)
        hidden : updated hidden state
        """
        out, hidden = self.lstm(x, hidden)
        pred = self.fc(out)
        return pred, hidden


class LSTMPredictor(nn.Module):
    """Full encoder-decoder LSTM predictor for exoskeleton motion."""

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 256,
        output_dim: int = 15,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        predict_frames: int = 1,
    ) -> None:
        super().__init__()
        self.predict_frames = predict_frames
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.decoder = LSTMDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        src : (batch, seq_len, input_dim) – IMU history window

        Returns
        -------
        preds : (batch, predict_frames, output_dim)
        """
        _, hidden = self.encoder(src)

        # Seed the decoder with the last observed input frame
        decoder_input = src[:, -1:, :]  # (batch, 1, input_dim)

        preds = []
        for _ in range(self.predict_frames):
            pred, hidden = self.decoder(decoder_input, hidden)
            preds.append(pred)
            # Use zeros as next decoder input (open-loop inference)
            decoder_input = torch.zeros_like(decoder_input)

        return torch.cat(preds, dim=1)  # (batch, predict_frames, output_dim)

    @classmethod
    def from_config(cls, cfg: dict) -> "LSTMPredictor":
        """Build from a config dict (e.g. loaded from YAML)."""
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]["lstm"]
        return cls(
            input_dim=data_cfg["input_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            output_dim=data_cfg["output_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            bidirectional=model_cfg.get("bidirectional", False),
            predict_frames=data_cfg.get("predict_frames", 1),
        )
