from base import BaseModel
import torch
import torch.nn as nn

from model.submodules import ConvLayer


class SimpleHSG(BaseModel):
    """
    Simple hidden-state generator for E2VID: shallow conv encoder + per-scale state heads.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_bins = int(config.get("num_bins", 5))
        self.num_encoders = int(config.get("num_encoders", 4))
        self.base_num_channels = int(config.get("base_num_channels", 32))
        self.norm = config.get("norm", None)
        self.recurrent_block_type = str(config.get("recurrent_block_type", "convlstm"))
        if self.recurrent_block_type not in ["convlstm", "convgru"]:
            raise ValueError(f"Unsupported recurrent_block_type: {self.recurrent_block_type}")

        self.head = ConvLayer(
            self.num_bins,
            self.base_num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=self.norm,
        )

        self.encoders = nn.ModuleList()
        self.state_h = nn.ModuleList()
        self.state_c = nn.ModuleList() if self.recurrent_block_type == "convlstm" else None

        in_ch = self.base_num_channels
        for i in range(self.num_encoders):
            out_ch = self.base_num_channels * (2 ** (i + 1))
            self.encoders.append(
                ConvLayer(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm=self.norm,
                )
            )
            self.state_h.append(nn.Conv2d(out_ch, out_ch, kernel_size=1))
            if self.state_c is not None:
                self.state_c.append(nn.Conv2d(out_ch, out_ch, kernel_size=1))
            in_ch = out_ch

        self.img_head = nn.Sequential(
            ConvLayer(
                self.base_num_channels,
                self.base_num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=self.norm,
            ),
            nn.Conv2d(self.base_num_channels, 1, kernel_size=1),
        )

    def forward(self, event_tensor, prev_states=None):
        x = self.head(event_tensor)
        img_pred = torch.sigmoid(self.img_head(x))

        states = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            h = self.state_h[i](x)
            if self.state_c is not None:
                c = self.state_c[i](x)
                states.append((h, c))
            else:
                states.append(h)

        return img_pred, states
