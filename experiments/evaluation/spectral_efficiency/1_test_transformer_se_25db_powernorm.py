import os
import csv
import math
import random
import numpy as np
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ============================================================
# 1. Transformer 模型结构
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TransformerRecon(nn.Module):
    """
    输入:
        x: (B, 2, 64, 32 * history_tti)

    输出:
        pred: (B, 2, 64, 32 * pred_tti)

    其中:
        64 = 子载波维度
        32 = 天线维度
    """

    def __init__(
        self,
        in_ch=2,
        H=64,
        history_tti=8,
        pred_tti=1,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        norm_first=True,
    ):
        super().__init__()

        assert history_tti >= 1, "history_tti must be >= 1"
        assert pred_tti >= 1, "pred_tti must be >= 1"

        self.in_ch = in_ch
        self.H = H
        self.history_tti = history_tti
        self.pred_tti = pred_tti

        self.W_in = 32 * history_tti
        self.W_out = 32 * pred_tti
        self.in_feat = in_ch * H

        self.input_proj = nn.Linear(self.in_feat, d_model)

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=self.W_in,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.reduce_len = nn.AdaptiveAvgPool1d(self.W_out)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.in_feat),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.0)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: (B, 2, 64, 32 * history_tti)
        """

        B, C, H, W = x.shape

        assert (C, H, W) == (self.in_ch, self.H, self.W_in), (
            f"expect (B,{self.in_ch},{self.H},{self.W_in}), got {tuple(x.shape)}"
        )

        # (B, 2, 64, W_in) -> (B, W_in, 2, 64) -> (B, W_in, 128)
        seq = x.permute(0, 3, 1, 2).contiguous()
        seq = seq.view(B, self.W_in, C * H)

        seq = self.input_proj(seq)
        seq = self.pos_encoding(seq)

        seq_out = self.encoder(seq)

        # (B, W_in, d_model) -> (B, d_model, W_in)
        seq_out = seq_out.transpose(1, 2)

        # 压缩到 W_out = 32 * pred_tti
        seq_out = self.reduce_len(seq_out)

        # (B, d_model, W_out) -> (B, W_out, d_model)
        seq_out = seq_out.transpose(1, 2).contiguous()

        step_feat = self.head(seq_out)

        step_feat = step_feat.view(
            B,
            self.W_out,
            self.in_ch,
            self.H,
        )

        pred = step_feat.permute(0, 2, 3, 1).contiguous()

        return pred


# ============================================================
# 2. 配置
# ============================================================

@dataclass
class Config:
    num2 = 20

    # data_dir: str = (
    #     "./"
    #     "multi_E_multi_dB/multi_e_CSI_data/"
    #     f"{num2}dB_zscore_norm_8to1_dataset"
    # )

    data_dir: str = (
        fr"./data/spectral_efficiency/{num2}dB_zscore_norm_8to1_dataset"
    )

    

    history_tti: int = 8
    pred_tti: int = 1

    batch_size: int = 128
    num_workers: int = 1

    amp: bool = True
    seed: int = 52
    device: str = "cuda"

    # Transformer 参数，如果 checkpoint 里有 config，会自动覆盖
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    norm_first: bool = True


cfg = Config()


# ============================================================
# 3. 测试配置
# ============================================================

SE_SNR_DB = 20
num3 = 90

# 改成你的 Transformer best.pt 路径
CKPT_PATH = "./checkpoints/spectral_efficiency/outputs_tsfm_20dB_8to1_1/8to1_2026-06-03_11-10-11/best.pt"

SAVE_CSV = fr"./result_0605_multi_SE/test_transformer_paper_style_SE_{SE_SNR_DB}dB_denorm_powernorm_results.csv"

# 你的 Z-score 参数目录
ZSCORE_STATS_DIR = fr"./Zscore_original/{SE_SNR_DB}dB_svd0{num3}_norm"

MEAN_REAL_CANDIDATES = [
    fr"global_mean_real_after_svd0{num3}.npy",
]

MEAN_IMAG_CANDIDATES = [
    fr"global_mean_imag_after_svd0{num3}.npy",
]

STD_REAL_CANDIDATES = [
    fr"global_std_real_after_svd0{num3}.npy",
]

STD_IMAG_CANDIDATES = [
    fr"global_std_imag_after_svd0{num3}.npy",
]

# 数据维度:
#   (B, 2, 64, 32 * pred_tti)
#
# 含义:
#   64 = 子载波维度
#   32 = 天线维度
NUM_SUBCARRIERS = 64
NUM_ANTENNAS = 32

# mean: 对 pred_tti 和 64 个子载波取平均
# sum : 对 64 个子载波求和，对 pred_tti 取平均
SE_REDUCTION = "mean"

DO_CHANNEL_POWER_NORM = True


# ============================================================
# 4. 基础工具函数
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str):
    if device_cfg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA 不可用，自动切换到 CPU。")
            return torch.device("cpu")

        if device_cfg == "cuda":
            return torch.device("cuda")

        idx = int(device_cfg.split(":")[1])

        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device '{device_cfg}', "
                f"but only {torch.cuda.device_count()} CUDA device(s) are available."
            )

        return torch.device(device_cfg)

    if device_cfg == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device config: {device_cfg}")


def make_autocast_ctx(use_cuda: bool, amp_enabled: bool):
    amp_enabled = bool(use_cuda and amp_enabled)

    def autocast_ctx():
        if not amp_enabled:
            return nullcontext()

        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast(enabled=True)

    return autocast_ctx


def nmse_to_db(x: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(x), eps))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


# ============================================================
# 5. 数据加载
# ============================================================

def complex_to_2ch(x: np.ndarray):
    """
    输入:
        x: (N, 64, 32*k), complex

    输出:
        x: (N, 2, 64, 32*k), float32
    """

    assert x.ndim == 3, f"expect 3D complex array, got shape={x.shape}"

    real = np.real(x)
    imag = np.imag(x)

    x = np.stack([real, imag], axis=1)

    return x.astype(np.float32)


def array_to_2ch(x: np.ndarray, name: str):
    """
    统一转成:
        (N, 2, 64, W)
    """

    if x.ndim == 4:
        if x.shape[1] == 2:
            return x.astype(np.float32)

        if x.shape[-1] == 2:
            x = np.transpose(x, (0, 3, 1, 2))
            return x.astype(np.float32)

        raise ValueError(f"{name} 是 4D，但无法识别通道维度，shape={x.shape}")

    if x.ndim == 3:
        if np.iscomplexobj(x):
            return complex_to_2ch(x)

        zeros = np.zeros_like(x)
        x = np.stack([x, zeros], axis=1)
        return x.astype(np.float32)

    raise ValueError(f"{name} 维度错误，期望 3D 或 4D，实际 shape={x.shape}")


def load_test_data_only(data_dir: str):
    test_inputs_path = os.path.join(data_dir, "test_inputs.npy")
    test_labels_path = os.path.join(data_dir, "test_labels.npy")

    for path in [test_inputs_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到测试数据文件: {path}")

    Xte = np.load(test_inputs_path)
    Yte = np.load(test_labels_path)

    print("\n" + "=" * 100)
    print("加载 Transformer 测试集")
    print(f"Data dir       : {data_dir}")
    print(f"Test inputs    : {test_inputs_path}")
    print(f"Test labels    : {test_labels_path}")
    print(f"Raw Xte        : {Xte.shape}, {Xte.dtype}")
    print(f"Raw Yte        : {Yte.shape}, {Yte.dtype}")
    print("=" * 100)

    Xte = array_to_2ch(Xte, "test_inputs.npy")
    Yte = array_to_2ch(Yte, "test_labels.npy")

    Xte = torch.from_numpy(Xte).float()
    Yte = torch.from_numpy(Yte).float()

    expected_x_shape = (2, NUM_SUBCARRIERS, NUM_ANTENNAS * cfg.history_tti)
    expected_y_shape = (2, NUM_SUBCARRIERS, NUM_ANTENNAS * cfg.pred_tti)

    assert Xte.shape[1:] == expected_x_shape, (
        f"test input shape error: {Xte.shape[1:]}, expected {expected_x_shape}"
    )

    assert Yte.shape[1:] == expected_y_shape, (
        f"test label shape error: {Yte.shape[1:]}, expected {expected_y_shape}"
    )

    print(f"Processed Xte  : {Xte.shape}, {Xte.dtype}")
    print(f"Processed Yte  : {Yte.shape}, {Yte.dtype}")

    return Xte, Yte


def make_test_loader(Xte, Yte, use_cuda: bool):
    test_ds = TensorDataset(Xte, Yte)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    return test_loader, len(test_ds)


# ============================================================
# 6. checkpoint 加载与模型构建
# ============================================================

def load_checkpoint_dict(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    return ckpt


def update_cfg_from_checkpoint(ckpt):
    """
    如果 checkpoint 里保存了 config，则优先使用训练时的模型超参数。
    """

    if not isinstance(ckpt, dict):
        return

    ckpt_cfg = ckpt.get("config", None)

    if not isinstance(ckpt_cfg, dict):
        return

    for name in [
        "history_tti",
        "pred_tti",
        "batch_size",
        "num_workers",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "norm_first",
    ]:
        if name in ckpt_cfg:
            setattr(cfg, name, ckpt_cfg[name])

    print("\n" + "=" * 100)
    print("已从 checkpoint config 更新模型相关参数")
    print(f"history_tti     : {cfg.history_tti}")
    print(f"pred_tti        : {cfg.pred_tti}")
    print(f"batch_size      : {cfg.batch_size}")
    print(f"num_workers     : {cfg.num_workers}")
    print(f"d_model         : {cfg.d_model}")
    print(f"nhead           : {cfg.nhead}")
    print(f"num_layers      : {cfg.num_layers}")
    print(f"dim_feedforward : {cfg.dim_feedforward}")
    print(f"dropout         : {cfg.dropout}")
    print(f"norm_first      : {cfg.norm_first}")
    print("=" * 100)


def build_model():
    model = TransformerRecon(
        in_ch=2,
        H=NUM_SUBCARRIERS,
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        norm_first=cfg.norm_first,
    )

    return model


def load_model_weights(model, ckpt, device):
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("成功加载 checkpoint 中的 ['model'] 权重")
    else:
        model.load_state_dict(ckpt)
        print("成功加载纯 state_dict 权重")

    model.to(device)
    model.eval()

    return model


# ============================================================
# 7. Z-score 参数加载与反标准化
# ============================================================

def find_existing_file(stats_dir: str, candidates):
    for name in candidates:
        path = os.path.join(stats_dir, name)

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"在目录 {stats_dir} 中找不到候选文件: {candidates}"
    )


def load_zscore_stats(stats_dir: str):
    paths = {
        "mean_real": find_existing_file(stats_dir, MEAN_REAL_CANDIDATES),
        "mean_imag": find_existing_file(stats_dir, MEAN_IMAG_CANDIDATES),
        "std_real": find_existing_file(stats_dir, STD_REAL_CANDIDATES),
        "std_imag": find_existing_file(stats_dir, STD_IMAG_CANDIDATES),
    }

    stats = {
        key: np.load(path)
        for key, path in paths.items()
    }

    print("\n" + "=" * 100)
    print("加载 Z-score 反标准化参数")
    print(f"Stats dir: {stats_dir}")

    for key, path in paths.items():
        value = stats[key]
        print(
            f"{key:10s}: {path}\n"
            f"            shape={value.shape}, dtype={value.dtype}, "
            f"min={np.min(value):.6e}, max={np.max(value):.6e}, "
            f"mean={np.mean(value):.6e}, abs_mean={np.mean(np.abs(value)):.6e}"
        )

    print("=" * 100)

    return stats


def _prepare_stat_tensor(stat_np, target_real_tensor: torch.Tensor, name: str):
    """
    把 mean/std 的 numpy 数组整理成可以和:
        target_real_tensor: (B, 64, W)
    广播的 torch tensor。
    """

    device = target_real_tensor.device
    dtype = target_real_tensor.dtype

    B, K, W = target_real_tensor.shape

    stat = torch.as_tensor(stat_np, device=device, dtype=dtype)
    stat = stat.squeeze()

    if stat.dim() == 0:
        return stat

    if stat.dim() == 1:
        if stat.numel() == K:
            return stat.view(1, K, 1)

        if stat.numel() == W:
            return stat.view(1, 1, W)

        if stat.numel() == NUM_ANTENNAS and W % NUM_ANTENNAS == 0:
            repeat_t = W // NUM_ANTENNAS
            return stat.view(1, 1, NUM_ANTENNAS).repeat(1, 1, repeat_t)

        raise ValueError(
            f"{name} 是 1D，但长度 {stat.numel()} 无法匹配目标 shape={(B, K, W)}"
        )

    if stat.dim() == 2:
        h, w = stat.shape

        if h == K and w == W:
            return stat.view(1, K, W)

        if h == K and w == NUM_ANTENNAS and W % NUM_ANTENNAS == 0:
            repeat_t = W // NUM_ANTENNAS
            return stat.view(1, K, NUM_ANTENNAS).repeat(1, 1, repeat_t)

        if h == K and w == 1:
            return stat.view(1, K, 1)

        if h == 1 and w == W:
            return stat.view(1, 1, W)

        if h == 1 and w == NUM_ANTENNAS and W % NUM_ANTENNAS == 0:
            repeat_t = W // NUM_ANTENNAS
            return stat.view(1, 1, NUM_ANTENNAS).repeat(1, 1, repeat_t)

        raise ValueError(
            f"{name} 是 2D，shape={tuple(stat.shape)} 无法匹配目标 shape={(B, K, W)}"
        )

    if stat.dim() == 3:
        if stat.shape[-2:] == (K, W):
            return stat.reshape(1, K, W)

        if stat.shape[-2:] == (K, NUM_ANTENNAS) and W % NUM_ANTENNAS == 0:
            repeat_t = W // NUM_ANTENNAS
            return stat.reshape(1, K, NUM_ANTENNAS).repeat(1, 1, repeat_t)

        raise ValueError(
            f"{name} 是 3D，shape={tuple(stat.shape)} 无法匹配目标 shape={(B, K, W)}"
        )

    raise ValueError(f"{name} 维度过多: shape={tuple(stat.shape)}")


@torch.no_grad()
def inverse_zscore_2ch(x_norm_2ch: torch.Tensor, stats: dict):
    """
    输入:
        x_norm_2ch: (B, 2, 64, 32 * pred_tti)

    输出:
        x_raw_2ch: (B, 2, 64, 32 * pred_tti)
    """

    assert x_norm_2ch.dim() == 4, f"x_norm_2ch 维度错误: {x_norm_2ch.shape}"
    assert x_norm_2ch.shape[1] == 2, f"通道维应该为 2，实际是 {x_norm_2ch.shape[1]}"
    assert x_norm_2ch.shape[2] == NUM_SUBCARRIERS, (
        f"子载波维应该为 {NUM_SUBCARRIERS}，实际是 {x_norm_2ch.shape[2]}"
    )

    real_norm = x_norm_2ch[:, 0, :, :].float()
    imag_norm = x_norm_2ch[:, 1, :, :].float()

    mean_real = _prepare_stat_tensor(stats["mean_real"], real_norm, "mean_real")
    mean_imag = _prepare_stat_tensor(stats["mean_imag"], imag_norm, "mean_imag")
    std_real = _prepare_stat_tensor(stats["std_real"], real_norm, "std_real")
    std_imag = _prepare_stat_tensor(stats["std_imag"], imag_norm, "std_imag")

    real_raw = real_norm * std_real + mean_real
    imag_raw = imag_norm * std_imag + mean_imag

    x_raw_2ch = torch.stack([real_raw, imag_raw], dim=1)

    return x_raw_2ch


# ============================================================
# 8. NMSE、功率统计、信道功率归一化
# ============================================================

def nmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12):
    diff = (pred - target).reshape(pred.size(0), -1)
    tgt = target.reshape(target.size(0), -1)

    nmse = diff.pow(2).sum(dim=1) / (tgt.pow(2).sum(dim=1) + eps)

    return nmse.mean()


@torch.no_grad()
def csi_power_mean(x_2ch: torch.Tensor):
    real = x_2ch[:, 0].float()
    imag = x_2ch[:, 1].float()

    power = real.pow(2) + imag.pow(2)

    return power.mean().item()


@torch.no_grad()
def csi_power_sum_count(x_2ch: torch.Tensor):
    real = x_2ch[:, 0].float()
    imag = x_2ch[:, 1].float()

    power = real.pow(2) + imag.pow(2)

    return power.sum().item(), power.numel()


@torch.no_grad()
def estimate_global_true_raw_power(test_loader, device, zscore_stats):
    use_cuda = device.type == "cuda"

    power_sum = 0.0
    power_count = 0

    pbar = tqdm(
        test_loader,
        desc="Estimating global true raw CSI power",
        ncols=120,
    )

    for _, y_true_norm in pbar:
        yb_norm = y_true_norm.to(device, non_blocking=use_cuda)
        yb_raw = inverse_zscore_2ch(yb_norm, zscore_stats)

        ps, pc = csi_power_sum_count(yb_raw)

        power_sum += ps
        power_count += pc

    global_power = power_sum / max(power_count, 1)
    power_scale = math.sqrt(max(global_power, 1e-30))

    print("\n" + "=" * 100)
    print("测试集真实 raw CSI 全局功率估计")
    print(f"global_true_raw_element_power : {global_power:.6e}")
    print(f"power_scale = sqrt(power)     : {power_scale:.6e}")
    print("后续执行: H_scaled = H_raw / power_scale")
    print("=" * 100)

    return global_power, power_scale


@torch.no_grad()
def apply_channel_power_norm(x_raw_2ch: torch.Tensor, power_scale: float):
    if not DO_CHANNEL_POWER_NORM:
        return x_raw_2ch

    return x_raw_2ch / float(power_scale)


# ============================================================
# 9. paper-style SE 计算
# ============================================================

@torch.no_grad()
def split_2ch_to_complex_tti(x_2ch: torch.Tensor):
    """
    输入:
        x_2ch: (B, 2, 64, 32 * pred_tti)

    输出:
        H: (B, pred_tti, 64, 32)
           即 (B, T, K, Nt)

    其中:
        K  = 64 个子载波
        Nt = 32 根天线
    """

    assert x_2ch.dim() == 4, f"x_2ch 维度错误: {x_2ch.shape}"
    assert x_2ch.shape[1] == 2, f"通道维应该为 2，实际是 {x_2ch.shape[1]}"
    assert x_2ch.shape[2] == NUM_SUBCARRIERS, (
        f"子载波维应该为 {NUM_SUBCARRIERS}，实际是 {x_2ch.shape[2]}"
    )

    B, _, K, W = x_2ch.shape

    assert W % NUM_ANTENNAS == 0, (
        f"W={W} 不能被天线数 {NUM_ANTENNAS} 整除，无法按 TTI 切分"
    )

    pred_tti = W // NUM_ANTENNAS

    real = x_2ch[:, 0, :, :].float()
    imag = x_2ch[:, 1, :, :].float()

    H_complex = torch.complex(real, imag)

    H_complex = H_complex.reshape(B, K, pred_tti, NUM_ANTENNAS)
    H_complex = H_complex.permute(0, 2, 1, 3).contiguous()

    return H_complex


@torch.no_grad()
def calc_paper_style_se(
    pred_2ch: torch.Tensor,
    true_2ch: torch.Tensor,
    snr_db: float = 25.0,
    reduction: str = "mean",
    eps: float = 1e-12,
):
    """
    论文式 SE:

        w_hat_k = h_hat_k / ||h_hat_k||
        SE_k = log2(1 + SNR * |h_true_k^H w_hat_k|^2)

    输入应为:
        反标准化 + 信道功率归一化后的 CSI
    """

    assert reduction in ["mean", "sum"], "reduction 必须是 'mean' 或 'sum'"

    H_hat = split_2ch_to_complex_tti(pred_2ch)
    H_true = split_2ch_to_complex_tti(true_2ch)

    snr_linear = 10.0 ** (snr_db / 10.0)

    norm = torch.sqrt(
        torch.sum(torch.abs(H_hat) ** 2, dim=3, keepdim=True) + eps
    )

    W_hat = H_hat / norm

    effective_gain = torch.sum(
        torch.conj(H_true) * W_hat,
        dim=3,
    )

    se_per_subcarrier = torch.log2(
        1.0 + snr_linear * torch.abs(effective_gain) ** 2
    )

    if reduction == "mean":
        se_per_sample = se_per_subcarrier.mean(dim=(1, 2))
    else:
        se_per_sample = se_per_subcarrier.sum(dim=2).mean(dim=1)

    return se_per_sample


# ============================================================
# 10. Transformer SE 测试评估
# ============================================================

@torch.no_grad()
def evaluate_transformer_paper_style_se(
    model,
    test_loader,
    device,
    zscore_stats,
    power_scale: float,
):
    model.eval()

    use_cuda = device.type == "cuda"
    autocast_ctx = make_autocast_ctx(use_cuda, cfg.amp)

    pred_se_sum = 0.0
    pred_se_count = 0

    perfect_se_sum = 0.0
    perfect_se_count = 0

    nmse_norm_sum = 0.0
    nmse_raw_sum = 0.0
    nmse_count = 0

    pred_raw_power_sum = 0.0
    true_raw_power_sum = 0.0

    pred_scaled_power_sum = 0.0
    true_scaled_power_sum = 0.0

    power_count = 0

    csv_rows = []

    pbar = tqdm(
        test_loader,
        desc=f"Testing Transformer denorm + powernorm paper-style SE at {SE_SNR_DB} dB",
        ncols=170,
    )

    for batch_idx, (x_input, y_true_norm) in enumerate(pbar):
        xb = x_input.to(device, non_blocking=use_cuda)
        yb_norm = y_true_norm.to(device, non_blocking=use_cuda)

        yb_raw = inverse_zscore_2ch(yb_norm, zscore_stats)
        yb_scaled = apply_channel_power_norm(yb_raw, power_scale)

        se_perfect = calc_paper_style_se(
            pred_2ch=yb_scaled,
            true_2ch=yb_scaled,
            snr_db=SE_SNR_DB,
            reduction=SE_REDUCTION,
        )

        perfect_se_batch_mean = se_perfect.mean().item()
        perfect_se_sum += se_perfect.sum().item()
        perfect_se_count += se_perfect.numel()

        true_raw_power_batch = csi_power_mean(yb_raw)
        true_scaled_power_batch = csi_power_mean(yb_scaled)

        with autocast_ctx():
            pred_norm = model(xb)

        pred_raw = inverse_zscore_2ch(pred_norm, zscore_stats)
        pred_scaled = apply_channel_power_norm(pred_raw, power_scale)

        se_pred = calc_paper_style_se(
            pred_2ch=pred_scaled,
            true_2ch=yb_scaled,
            snr_db=SE_SNR_DB,
            reduction=SE_REDUCTION,
        )

        pred_se_batch_mean = se_pred.mean().item()
        pred_se_sum += se_pred.sum().item()
        pred_se_count += se_pred.numel()

        loss_norm = nmse_loss(pred_norm, yb_norm)
        loss_raw = nmse_loss(pred_raw, yb_raw)

        bs = xb.size(0)

        nmse_norm_sum += loss_norm.item() * bs
        nmse_raw_sum += loss_raw.item() * bs
        nmse_count += bs

        pred_raw_power_batch = csi_power_mean(pred_raw)
        pred_scaled_power_batch = csi_power_mean(pred_scaled)

        pred_raw_power_sum += pred_raw_power_batch
        true_raw_power_sum += true_raw_power_batch

        pred_scaled_power_sum += pred_scaled_power_batch
        true_scaled_power_sum += true_scaled_power_batch

        power_count += 1

        se_gap_batch = perfect_se_batch_mean - pred_se_batch_mean
        se_ratio_batch = pred_se_batch_mean / max(perfect_se_batch_mean, 1e-12)

        raw_power_ratio_batch = pred_raw_power_batch / max(true_raw_power_batch, 1e-12)
        scaled_power_ratio_batch = pred_scaled_power_batch / max(true_scaled_power_batch, 1e-12)

        csv_rows.append([
            batch_idx,
            pred_se_batch_mean,
            perfect_se_batch_mean,
            se_gap_batch,
            se_ratio_batch,
            loss_norm.item(),
            nmse_to_db(loss_norm.item()),
            loss_raw.item(),
            nmse_to_db(loss_raw.item()),
            pred_raw_power_batch,
            true_raw_power_batch,
            raw_power_ratio_batch,
            pred_scaled_power_batch,
            true_scaled_power_batch,
            scaled_power_ratio_batch,
        ])

        pbar.set_postfix(
            pred_SE=f"{pred_se_batch_mean:.4f}",
            perfect_SE=f"{perfect_se_batch_mean:.4f}",
            ratio=f"{se_ratio_batch:.4f}",
            norm_nmse_db=f"{nmse_to_db(loss_norm.item()):.2f}",
            raw_nmse_db=f"{nmse_to_db(loss_raw.item()):.2f}",
            pwr_scaled=f"{true_scaled_power_batch:.3f}",
        )

    pred_se_mean = pred_se_sum / max(pred_se_count, 1)
    perfect_se_mean = perfect_se_sum / max(perfect_se_count, 1)

    nmse_norm_mean = nmse_norm_sum / max(nmse_count, 1)
    nmse_raw_mean = nmse_raw_sum / max(nmse_count, 1)

    nmse_norm_db = nmse_to_db(nmse_norm_mean)
    nmse_raw_db = nmse_to_db(nmse_raw_mean)

    se_gap = perfect_se_mean - pred_se_mean
    se_ratio = pred_se_mean / max(perfect_se_mean, 1e-12)

    pred_raw_power_mean = pred_raw_power_sum / max(power_count, 1)
    true_raw_power_mean = true_raw_power_sum / max(power_count, 1)

    pred_scaled_power_mean = pred_scaled_power_sum / max(power_count, 1)
    true_scaled_power_mean = true_scaled_power_sum / max(power_count, 1)

    raw_power_ratio = pred_raw_power_mean / max(true_raw_power_mean, 1e-12)
    scaled_power_ratio = pred_scaled_power_mean / max(true_scaled_power_mean, 1e-12)

    result = {
        "pred_se_mean": pred_se_mean,
        "perfect_se_mean": perfect_se_mean,
        "se_gap": se_gap,
        "se_ratio": se_ratio,

        "nmse_norm_mean": nmse_norm_mean,
        "nmse_norm_db": nmse_norm_db,
        "nmse_raw_mean": nmse_raw_mean,
        "nmse_raw_db": nmse_raw_db,

        "pred_raw_power_mean": pred_raw_power_mean,
        "true_raw_power_mean": true_raw_power_mean,
        "raw_power_ratio": raw_power_ratio,

        "pred_scaled_power_mean": pred_scaled_power_mean,
        "true_scaled_power_mean": true_scaled_power_mean,
        "scaled_power_ratio": scaled_power_ratio,

        "csv_rows": csv_rows,
    }

    return result


# ============================================================
# 11. 主函数
# ============================================================

def main():
    set_seed(cfg.seed)

    print("\n" + "=" * 100)
    print("Transformer 测试：25 dB Paper-style SE，反 Z-score + 信道功率归一化")
    print(f"Checkpoint             : {CKPT_PATH}")
    print(f"Data dir               : {cfg.data_dir}")
    print(f"Zscore stats           : {ZSCORE_STATS_DIR}")
    print("=" * 100)

    ckpt = load_checkpoint_dict(CKPT_PATH)

    update_cfg_from_checkpoint(ckpt)

    device = resolve_device(cfg.device)
    use_cuda = device.type == "cuda"

    print("\n" + "=" * 100)
    print("测试配置")
    print(f"Device                 : {device}")
    print(f"SNR                    : {SE_SNR_DB} dB")
    print(f"history_tti            : {cfg.history_tti}")
    print(f"pred_tti               : {cfg.pred_tti}")
    print(f"d_model                : {cfg.d_model}")
    print(f"nhead                  : {cfg.nhead}")
    print(f"num_layers             : {cfg.num_layers}")
    print(f"dim_feedforward        : {cfg.dim_feedforward}")
    print(f"dropout                : {cfg.dropout}")
    print(f"norm_first             : {cfg.norm_first}")
    print(f"NUM_SUBCARRIERS        : {NUM_SUBCARRIERS}")
    print(f"NUM_ANTENNAS           : {NUM_ANTENNAS}")
    print(f"SE_REDUCTION           : {SE_REDUCTION}")
    print(f"DO_CHANNEL_POWER_NORM  : {DO_CHANNEL_POWER_NORM}")
    print("=" * 100)

    zscore_stats = load_zscore_stats(ZSCORE_STATS_DIR)

    Xte, Yte = load_test_data_only(cfg.data_dir)

    test_loader, test_size = make_test_loader(
        Xte,
        Yte,
        use_cuda=use_cuda,
    )

    print(f"Test samples: {test_size}")

    global_true_raw_power, power_scale = estimate_global_true_raw_power(
        test_loader=test_loader,
        device=device,
        zscore_stats=zscore_stats,
    )

    model = build_model()

    total_params, trainable_params = count_parameters(model)

    print("\n" + "=" * 100)
    print("模型信息")
    print("Model type: TransformerRecon")
    print(f"Parameters total     : {total_params:,}")
    print(f"Parameters trainable : {trainable_params:,}")
    print("=" * 100)

    model = load_model_weights(
        model=model,
        ckpt=ckpt,
        device=device,
    )

    result = evaluate_transformer_paper_style_se(
        model=model,
        test_loader=test_loader,
        device=device,
        zscore_stats=zscore_stats,
        power_scale=power_scale,
    )

    print("\n" + "=" * 100)
    print("Transformer 25 dB 测试集 Paper-style SE 结果：反 Z-score + 信道功率归一化")
    print(f"global true raw element power    : {global_true_raw_power:.6e}")
    print(f"power scale sqrt(power)          : {power_scale:.6e}")
    print("-" * 100)
    print(f"预测 CSI 预编码 SE               : {result['pred_se_mean']:.6f} bit/s/Hz")
    print(f"Perfect CSI SE                   : {result['perfect_se_mean']:.6f} bit/s/Hz")
    print(f"SE gap, perfect - pred           : {result['se_gap']:.6f} bit/s/Hz")
    print(f"SE ratio, pred / perfect         : {result['se_ratio']:.6f}")
    print("-" * 100)
    print(f"标准化域 NMSE                    : {result['nmse_norm_mean']:.6e}")
    print(f"标准化域 NMSE(dB)                : {result['nmse_norm_db']:.2f} dB")
    print(f"反标准化 raw 域 NMSE             : {result['nmse_raw_mean']:.6e}")
    print(f"反标准化 raw 域 NMSE(dB)         : {result['nmse_raw_db']:.2f} dB")
    print("-" * 100)
    print(f"预测 raw CSI 平均功率            : {result['pred_raw_power_mean']:.6e}")
    print(f"真实 raw CSI 平均功率            : {result['true_raw_power_mean']:.6e}")
    print(f"raw 功率比 pred / true           : {result['raw_power_ratio']:.6f}")
    print("-" * 100)
    print(f"预测 scaled CSI 平均功率         : {result['pred_scaled_power_mean']:.6e}")
    print(f"真实 scaled CSI 平均功率         : {result['true_scaled_power_mean']:.6e}")
    print(f"scaled 功率比 pred / true        : {result['scaled_power_ratio']:.6f}")
    print("=" * 100)

    save_dir = os.path.dirname(SAVE_CSV)

    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    with open(SAVE_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "batch_idx",
            "pred_precoding_se_mean_denorm_powernorm",
            "perfect_csi_se_mean_denorm_powernorm",
            "se_gap_perfect_minus_pred",
            "se_ratio_pred_over_perfect",
            "nmse_norm",
            "nmse_norm_db",
            "nmse_raw",
            "nmse_raw_db",
            "pred_raw_power",
            "true_raw_power",
            "raw_power_ratio_pred_over_true",
            "pred_scaled_power",
            "true_scaled_power",
            "scaled_power_ratio_pred_over_true",
        ])

        for row in result["csv_rows"]:
            writer.writerow(row)

    print(f"详细结果已保存到: {SAVE_CSV}")


if __name__ == "__main__":
    main()