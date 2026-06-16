import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


def nmse_and_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12):
    diff = (pred - target).reshape(pred.size(0), -1)
    tgt = target.reshape(target.size(0), -1)
    nmse = diff.pow(2).sum(dim=1) / (tgt.pow(2).sum(dim=1) + eps)
    nmse_mean = nmse.mean()
    nmse_db = 10.0 * torch.log10(torch.clamp(nmse_mean, min=eps))
    return nmse_mean, nmse_db


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 4):
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class ResBlock(nn.Module):
    def __init__(self, in_planes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        y = self.ca(y) * y
        return x + y


class LLM4CPCSIProject(nn.Module):
    """
    数据语义:
      单个 TTI: (2, 64, 32)
      其中:
        64 = 子载波数
        32 = 天线数

    输入:
      x_input: (B, 2, 64, 32 * history_tti)
      y_label: (B, 2, 64, 32 * pred_tti)

    输出:
      pred:    (B, 2, 64, 32 * pred_tti)
    """

    def __init__(
        self,
        history_tti: int = 8,
        pred_tti: int = 1,
        num_subcarriers: int = 64,
        num_antennas: int = 32,
        in_chans: int = 2,
        proj_ant: int = 4,
        gpt_type: str = "gpt2",
        gpt_layers: int = 8,
        d_model: int = 384,
        d_ff: int = 384,
        patch_size: int = 2,
        res_layers: int = 3,
        res_dim: int = 64,
        dropout: float = 0.1,
        freeze_gpt: bool = True,
        gpt_path: Optional[str] = None,
        local_files_only: bool = True,
    ):
        super().__init__()
        assert history_tti >= 1
        assert pred_tti >= 1
        assert history_tti % patch_size == 0, "history_tti must be divisible by patch_size"
        assert proj_ant >= 1

        self.history_tti = history_tti
        self.pred_tti = pred_tti
        self.num_subcarriers = num_subcarriers   # 64
        self.num_antennas = num_antennas         # 32
        self.in_chans = in_chans
        self.proj_ant = proj_ant
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_ff = d_ff

        self.step_feature_dim = in_chans * num_subcarriers * num_antennas
        self.enc_in = num_subcarriers * proj_ant
        self.seq_feature_dim = 2 * self.enc_in

        # 压缩的是天线维: 32 -> proj_ant
        self.antenna_proj = nn.Linear(num_antennas, proj_ant, bias=False)

        self.enc_embedding = DataEmbedding(self.seq_feature_dim, d_model, dropout)

        gpt_name_or_path = gpt_path if gpt_path is not None else gpt_type

        if gpt_type == "gpt2-medium":
            self.gpt2 = GPT2Model.from_pretrained(
                gpt_name_or_path,
                output_attentions=True,
                output_hidden_states=True,
                local_files_only=local_files_only,
            )
            self.gpt_dim = 1024
        elif gpt_type == "gpt2-large":
            self.gpt2 = GPT2Model.from_pretrained(
                gpt_name_or_path,
                output_attentions=True,
                output_hidden_states=True,
                local_files_only=local_files_only,
            )
            self.gpt_dim = 1280
        elif gpt_type == "gpt2-xl":
            self.gpt2 = GPT2Model.from_pretrained(
                gpt_name_or_path,
                output_attentions=True,
                output_hidden_states=True,
                local_files_only=local_files_only,
            )
            self.gpt_dim = 1600
        else:
            self.gpt2 = GPT2Model.from_pretrained(
                gpt_name_or_path,
                output_attentions=True,
                output_hidden_states=True,
                local_files_only=local_files_only,
            )
            self.gpt_dim = 768

        self.gpt2.h = self.gpt2.h[:gpt_layers]

        if freeze_gpt:
            for name, param in self.gpt2.named_parameters():
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        self.predict_linear_pre = nn.Linear(self.history_tti, self.history_tti)

        self.rb_freq = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        self.rb_delay = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))

        for _ in range(res_layers):
            self.rb_freq.append(ResBlock(res_dim))
            self.rb_delay.append(ResBlock(res_dim))

        self.rb_freq.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        self.rb_delay.append(nn.Conv2d(res_dim, 2, 3, 1, 1))

        # 直接回归完整一个 TTI 的 flattened CSI
        self.out_layer_dim = nn.Linear(d_ff, self.step_feature_dim)
        self.output_layer_time = nn.Linear(self.history_tti, self.pred_tti)

    def _reshape_input_to_complex_grid(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        (B, 2, 64, 32*T) -> (B, T, 64, 32) complex
        其中:
          64 = 子载波
          32 = 天线
        """
        b, _, sc, _ = x_input.shape
        x = x_input.view(b, self.in_chans, sc, self.history_tti, self.num_antennas)
        x = x.permute(0, 3, 1, 2, 4).contiguous()  # (B,T,2,64,32)
        return torch.complex(x[:, :, 0], x[:, :, 1])

    def _project_antennas(self, x_complex: torch.Tensor) -> torch.Tensor:
        """
        (B,T,64,32) complex -> (B,T,64,proj_ant) complex
        沿最后一维天线维做投影
        """
        xr = self.antenna_proj(x_complex.real)
        xi = self.antenna_proj(x_complex.imag)
        return torch.complex(xr, xi)

    def _complex_grid_to_sequence(self, x_complex: torch.Tensor) -> torch.Tensor:
        """
        (B,T,64,proj_ant) complex -> (B,T,2*K)
        K = 64 * proj_ant
        使用 real, imag 交错排列
        """
        x_ri = torch.stack([x_complex.real, x_complex.imag], dim=-1)
        return x_ri.reshape(x_complex.size(0), x_complex.size(1), -1)

    def _sequence_to_feature_map(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        (B,T,2*K) -> (B,2,T,K)
        """
        b, t, _ = x_seq.shape
        x = x_seq.view(b, t, self.enc_in, 2)
        return x.permute(0, 3, 1, 2).contiguous()

    def _sequence_to_csi(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        (B,pred_tti,4096) -> (B,2,64,32*pred_tti)
        """
        b = x_seq.size(0)
        x = x_seq.view(b, self.pred_tti, self.in_chans, self.num_subcarriers, self.num_antennas)
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        return x.view(b, self.in_chans, self.num_subcarriers, self.pred_tti * self.num_antennas)

    def forward(
        self,
        x_input: torch.Tensor,
        y_label: Optional[torch.Tensor] = None,
    ):
        # 频域低维特征
        x_complex = self._reshape_input_to_complex_grid(x_input)   # (B,T,64,32)
        x_proj_freq = self._project_antennas(x_complex)            # (B,T,64,proj_ant)
        x_seq_freq = self._complex_grid_to_sequence(x_proj_freq)   # (B,T,2*K)

        # 时延域特征: 沿子载波维 64 做 IFFT
        x_proj_delay = torch.fft.ifft(x_proj_freq, dim=2)
        x_seq_delay = self._complex_grid_to_sequence(x_proj_delay)

        # 与你前面一致: per-sample 归一化
        mean = x_seq_freq.mean(dim=(1, 2), keepdim=True)
        std = x_seq_freq.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)

        x_seq_freq = (x_seq_freq - mean) / std
        x_seq_delay = (x_seq_delay - mean) / std

        b, l, d = x_seq_freq.shape

        # delay branch
        x_enc_delay = x_seq_delay.reshape(b, l // self.patch_size, self.patch_size, d)
        x_enc_delay = self.patch_layer(x_enc_delay.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_delay = x_enc_delay.reshape(b, l, d)
        x_enc_delay = self._sequence_to_feature_map(x_enc_delay)
        x_enc_delay = self.rb_delay(x_enc_delay)

        # freq branch
        x_enc_freq = x_seq_freq.reshape(b, l // self.patch_size, self.patch_size, d)
        x_enc_freq = self.patch_layer(x_enc_freq.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_freq = x_enc_freq.reshape(b, l, d)
        x_enc_freq = self._sequence_to_feature_map(x_enc_freq)
        x_enc_freq = self.rb_freq(x_enc_freq)

        x_enc = x_enc_freq + x_enc_delay
        x_enc = x_enc.permute(0, 2, 3, 1).contiguous().view(b, l, d)

        enc_out = self.enc_embedding(x_enc)
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        if enc_out.shape[-1] < self.gpt_dim:
            pad_width = self.gpt_dim - enc_out.shape[-1]
            enc_out = F.pad(enc_out, (0, pad_width))
        else:
            enc_out = enc_out[..., : self.gpt_dim]

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, : self.d_ff]
        dec_out = self.out_layer_dim(dec_out)
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = dec_out * std + mean

        pred = self._sequence_to_csi(dec_out)

        if y_label is None:
            return pred

        nmse_mean, nmse_db = nmse_and_db(pred, y_label)
        loss = nmse_mean
        loss_dict: Dict[str, torch.Tensor] = {
            "nmse": nmse_mean,
            "nmse_db": nmse_db,
        }
        aux = torch.empty(0, device=x_input.device)
        return loss, pred, aux, loss_dict


def build_llm4cp_csi(history_tti: int = 8, pred_tti: int = 1, **kwargs):
    return LLM4CPCSIProject(
        history_tti=history_tti,
        pred_tti=pred_tti,
        num_subcarriers=64,
        num_antennas=32,
        in_chans=2,
        proj_ant=4,
        gpt_type="gpt2",
        gpt_layers=8,
        d_model=512,
        d_ff=512,
        patch_size=2,
        res_layers=2,
        res_dim=64,
        dropout=0.1,
        freeze_gpt=True,
        **kwargs,
    )
