import os
import math
import numpy as np
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 改成你第一段模型代码所在的文件名
# 例如如果第一段代码在 models_mae_natural_8tox.py 里，就用这一行
from models_mae_plain_original import build_mae_channel_plain_mae


# =========================
# 给 FLOPs 用的包装器：只返回 pred
# =========================
class MAEPredWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, y):
        _, pred, _, _ = self.model(x, y)
        return pred


# =========================
# 配置
# =========================
@dataclass
class Config:
    # 数据集文件夹，里面需要有 test_inputs.npy / test_labels.npy
    # 也兼容 test_input.npy / test_label.npy
    data_dir: str = r"./data/paper_metrics/None_svd_speed_test_8to1_10to100_step2_v1/all_speed_dataset"

    # 训练好的模型权重
    ckpt_path: str = r"./checkpoints/Original_MAE.pt"

    history_tti: int = 8
    pred_tti: int = 1
    keep_patches: int = 8

    batch_size: int = 128
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    eps: float = 1e-12
    seed: int = 2026

    save_per_sample: bool = True
    save_predictions: bool = False

    gpu_name_for_theory: str = "NVIDIA RTX 3090"
    gpu_tflops_fp32: float = 35.58


cfg = Config()


# =========================
# 工具函数
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_generator(device, seed: int):
    """
    你的 plain MAE 模型测试时仍会随机保留 patch。
    固定 generator 后，每次测试结果可复现。
    """
    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator()

    generator.manual_seed(seed)
    return generator


def resolve_device(device_cfg: str):
    if device_cfg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warning] CUDA 不可用，自动使用 CPU")
            return torch.device("cpu")

        if device_cfg == "cuda":
            return torch.device("cuda")

        idx = int(device_cfg.split(":")[1])
        if idx >= torch.cuda.device_count():
            print(f"[warning] 指定 {device_cfg}，但只有 {torch.cuda.device_count()} 张 GPU，自动使用 cuda:0")
            return torch.device("cuda:0")

        return torch.device(device_cfg)

    return torch.device("cpu")


def to_db(x: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(x), eps))


def format_count(num: float) -> str:
    if num >= 1e12:
        return f"{num / 1e12:.2f}T"
    if num >= 1e9:
        return f"{num / 1e9:.2f}G"
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    if num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


def find_existing_file(data_dir: str, names):
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"在文件夹 {data_dir} 中没有找到这些文件：{names}"
    )


def complex_to_2ch(x: np.ndarray):
    """
    complex 数据：
        输入: (N, 64, 32*T)
        输出: (N, 2, 64, 32*T)
    """
    assert x.ndim == 3, f"complex 数据应为 3D，即 (N,64,W)，但当前是 {x.shape}"

    real = np.real(x)
    imag = np.imag(x)

    x_2ch = np.stack([real, imag], axis=1)
    return x_2ch.astype(np.float32)


def normalize_data_shape(arr: np.ndarray, name: str, expected_width: int):
    """
    支持三种格式：
    1. complex: (N, 64, W)
    2. 实虚双通道: (N, 2, 64, W)
    3. 实虚最后一维: (N, 64, W, 2)
    """
    print(f"Raw {name}: shape={arr.shape}, dtype={arr.dtype}")

    if np.iscomplexobj(arr):
        arr = complex_to_2ch(arr)

    elif arr.ndim == 4:
        if arr.shape[1] == 2:
            arr = arr.astype(np.float32)
        elif arr.shape[-1] == 2:
            arr = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32)
        else:
            raise ValueError(
                f"{name} 是 4D，但不是 (N,2,64,W) 或 (N,64,W,2)，当前 shape={arr.shape}"
            )

    elif arr.ndim == 3:
        print(f"[warning] {name} 是实数 3D 数据，将自动补零虚部通道")
        arr = np.stack([arr, np.zeros_like(arr)], axis=1).astype(np.float32)

    else:
        raise ValueError(f"{name} 维度错误，当前 shape={arr.shape}")

    assert arr.shape[1] == 2, f"{name} 通道数应为 2，当前 shape={arr.shape}"
    assert arr.shape[2] == 64, f"{name} 高度应为 64，当前 shape={arr.shape}"
    assert arr.shape[3] == expected_width, (
        f"{name} 宽度错误，当前为 {arr.shape[3]}，应为 {expected_width}"
    )

    print(f"Used {name}: shape={arr.shape}, dtype={arr.dtype}")
    return arr


def load_test_data(data_dir: str, history_tti: int, pred_tti: int):
    x_path = find_existing_file(
        data_dir,
        ["test_inputs.npy", "test_input.npy", "X_test.npy", "test_X.npy"]
    )
    y_path = find_existing_file(
        data_dir,
        ["test_labels.npy", "test_label.npy", "Y_test.npy", "test_Y.npy"]
    )

    Xte = np.load(x_path)
    Yte = np.load(y_path)

    Xte = normalize_data_shape(
        Xte,
        name="Xte",
        expected_width=32 * history_tti,
    )

    Yte = normalize_data_shape(
        Yte,
        name="Yte",
        expected_width=32 * pred_tti,
    )

    Xte = torch.from_numpy(Xte).float()
    Yte = torch.from_numpy(Yte).float()

    return Xte, Yte


class TTIDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert len(self.X) == len(self.Y), (
            f"X 和 Y 样本数不一致：{len(self.X)} vs {len(self.Y)}"
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def make_loader(X, Y, batch_size, num_workers, use_cuda):
    dataset = TTIDataset(X, Y)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    return loader


def strip_prefix_if_present(state_dict, prefix: str):
    keys = list(state_dict.keys())

    if len(keys) == 0:
        return state_dict

    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def load_checkpoint(model: nn.Module, ckpt_path: str, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"模型权重不存在：{ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    state = strip_prefix_if_present(state, "module.")
    state = strip_prefix_if_present(state, "model.")

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing or unexpected:
        print("[warning] load_state_dict 不是完全匹配")
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
    else:
        print("Checkpoint loaded successfully.")

    if isinstance(ckpt, dict):
        for key in ["epoch", "best_loss", "best_nmse", "val_loss", "config"]:
            if key in ckpt:
                print(f"Checkpoint {key}: {ckpt[key]}")

    return ckpt


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def param_memory_mb(num_params: int):
    """
    按 32-bit float 估算参数存储开销。
    """
    return num_params * 4 / (1024 ** 2)


def get_ckpt_size_mb(path: str):
    return os.path.getsize(path) / (1024 ** 2)


def theoretical_time_from_flops(flops: float, gpu_tflops: float):
    if flops is None:
        return float("nan")

    return flops / (gpu_tflops * 1e12)


# =========================
# FLOPs
# =========================
def compute_flops(model, device, history_tti, pred_tti):
    try:
        from thop import profile
    except ImportError:
        print("[warning] 没有安装 thop，无法统计 FLOPs。安装命令：pip install thop")
        return None

    model.eval()
    wrapper = MAEPredWrapper(model).to(device)

    dummy_x = torch.randn(1, 2, 64, 32 * history_tti, device=device)
    dummy_y = torch.randn(1, 2, 64, 32 * pred_tti, device=device)

    flops, _ = profile(
        wrapper,
        inputs=(dummy_x, dummy_y),
        verbose=False,
    )

    return flops


# =========================
# 指标计算
# =========================
@torch.no_grad()
def evaluate_metrics(
    model,
    loader,
    device,
    generator,
    eps=1e-12,
    save_per_sample=True,
    save_predictions=False,
):
    model.eval()

    sse_global = 0.0
    power_global = 0.0

    total_elems = 0
    sum_y = 0.0
    sum_y2 = 0.0

    nmse_each = []
    evm_each = []
    r2_each = []

    pred_all = []
    label_all = []

    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # plain MAE 模型内部有随机 mask，测试时传入固定 generator
        _, pred, _, loss_dict = model(
            xb,
            yb,
            generator=generator,
        )

        y = yb.reshape(yb.size(0), -1)
        ph = pred.reshape(pred.size(0), -1)

        err = ph - y

        sse = torch.sum(err ** 2, dim=1)
        power = torch.sum(y ** 2, dim=1)

        sse_global += sse.sum().item()
        power_global += power.sum().item()

        sum_y += torch.sum(y).item()
        sum_y2 += torch.sum(y ** 2).item()
        total_elems += y.numel()

        if save_per_sample:
            nmse = sse / torch.clamp(power, min=eps)
            evm = torch.sqrt(nmse) * 100.0

            y_mean = y.mean(dim=1, keepdim=True)
            sst = torch.sum((y - y_mean) ** 2, dim=1)

            r2 = torch.where(
                sst > eps,
                1.0 - sse / sst,
                torch.tensor(float("nan"), device=device),
            )

            nmse_each.append(nmse.detach().cpu())
            evm_each.append(evm.detach().cpu())
            r2_each.append(r2.detach().cpu())

        if save_predictions:
            pred_all.append(pred.detach().cpu())
            label_all.append(yb.detach().cpu())

    global_nmse = sse_global / max(power_global, eps)
    global_nmse_db = 10.0 * math.log10(max(global_nmse, eps))
    global_evm = math.sqrt(global_nmse) * 100.0

    y_mean_global = sum_y / max(total_elems, 1)
    sst_global = sum_y2 - total_elems * (y_mean_global ** 2)

    if sst_global <= eps:
        global_r2 = float("nan")
    else:
        global_r2 = 1.0 - sse_global / sst_global

    results = {
        "global_nmse": global_nmse,
        "global_nmse_db": global_nmse_db,
        "global_r2": global_r2,
        "global_evm_percent": global_evm,
    }

    if save_per_sample:
        results["nmse_per_sample"] = torch.cat(nmse_each, dim=0).numpy()
        results["r2_per_sample"] = torch.cat(r2_each, dim=0).numpy()
        results["evm_per_sample"] = torch.cat(evm_each, dim=0).numpy()

    if save_predictions:
        results["pred_all"] = torch.cat(pred_all, dim=0).numpy()
        results["label_all"] = torch.cat(label_all, dim=0).numpy()

    return results


# =========================
# 保存结果
# =========================
def save_results(
    cfg,
    results,
    total_params,
    trainable_params,
    param_mem_mb,
    ckpt_size_mb,
    flops,
    theory_time,
    device,
):
    os.makedirs(cfg.data_dir, exist_ok=True)

    if cfg.save_per_sample:
        np.save(
            os.path.join(cfg.data_dir, "plain_original_mae_nmse_per_sample.npy"),
            results["nmse_per_sample"],
        )
        np.save(
            os.path.join(cfg.data_dir, "plain_original_mae_r2_per_sample.npy"),
            results["r2_per_sample"],
        )
        np.save(
            os.path.join(cfg.data_dir, "plain_original_mae_evm_per_sample.npy"),
            results["evm_per_sample"],
        )

    if cfg.save_predictions:
        np.save(
            os.path.join(cfg.data_dir, "plain_original_mae_predictions.npy"),
            results["pred_all"],
        )
        np.save(
            os.path.join(cfg.data_dir, "plain_original_mae_labels.npy"),
            results["label_all"],
        )

    summary_path = os.path.join(cfg.data_dir, "plain_original_mae_eval_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("========== Plain Original MAE Test Summary ==========\n")
        f.write(f"Config: {asdict(cfg)}\n")
        f.write(f"Device: {device}\n")

        if device.type == "cuda":
            f.write(f"Actual CUDA device: {torch.cuda.get_device_name(device)}\n")

        f.write("\n")
        f.write("========== Metrics ==========\n")
        f.write(f"Test NMSE: {results['global_nmse']:.6e}\n")
        f.write(f"Test NMSE(dB): {results['global_nmse_db']:.2f}\n")
        f.write(f"Test R2: {results['global_r2']:.6f}\n")
        f.write(f"Test EVM(%): {results['global_evm_percent']:.4f}\n")

        f.write("\n")
        f.write("========== Complexity ==========\n")
        f.write(f"Params total: {total_params} ({format_count(total_params)})\n")
        f.write(f"Params trainable: {trainable_params} ({format_count(trainable_params)})\n")
        f.write(f"Parameter memory(MB, fp32): {param_mem_mb:.2f}\n")
        f.write(f"Checkpoint size(MB): {ckpt_size_mb:.2f}\n")

        if flops is not None:
            f.write(f"FLOPs: {flops:.0f} ({format_count(flops)})\n")
            f.write(
                f"Theoretical time on {cfg.gpu_name_for_theory}: {theory_time:.6e} s\n"
            )

        if cfg.save_per_sample:
            valid_nmse = results["nmse_per_sample"][~np.isnan(results["nmse_per_sample"])]
            valid_r2 = results["r2_per_sample"][~np.isnan(results["r2_per_sample"])]
            valid_evm = results["evm_per_sample"][~np.isnan(results["evm_per_sample"])]

            f.write("\n")
            f.write("========== Per-sample Statistics ==========\n")

            if valid_nmse.size > 0:
                f.write(f"Per-sample NMSE mean: {valid_nmse.mean():.6e}\n")
                f.write(f"Per-sample NMSE std: {valid_nmse.std():.6e}\n")

            if valid_r2.size > 0:
                f.write(f"Per-sample R2 mean: {valid_r2.mean():.6f}\n")
                f.write(f"Per-sample R2 std: {valid_r2.std():.6f}\n")
                f.write(
                    f"Per-sample R2 NaN count: {results['r2_per_sample'].size - valid_r2.size}\n"
                )

            if valid_evm.size > 0:
                f.write(f"Per-sample EVM mean(%): {valid_evm.mean():.4f}\n")
                f.write(f"Per-sample EVM std(%): {valid_evm.std():.4f}\n")

    print(f"Summary saved to: {summary_path}")


# =========================
# 主函数
# =========================
def main():
    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    use_cuda = device.type == "cuda"
    generator = make_generator(device, cfg.seed)

    print("========== Config ==========")
    print(asdict(cfg))

    print()
    print("========== Device ==========")
    print("Device:", device)

    if use_cuda:
        print("Actual CUDA device:", torch.cuda.get_device_name(device))

    print()
    print("========== Load Data ==========")
    Xte, Yte = load_test_data(
        data_dir=cfg.data_dir,
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
    )

    loader = make_loader(
        Xte,
        Yte,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_cuda=use_cuda,
    )

    print()
    print("========== Build Model ==========")
    model = build_mae_channel_plain_mae(
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
        keep_patches=cfg.keep_patches,
    ).to(device)

    print(model.__class__.__name__)

    print()
    print("========== Load Checkpoint ==========")
    _ = load_checkpoint(
        model=model,
        ckpt_path=cfg.ckpt_path,
        device=device,
    )

    print()
    print("========== Evaluate ==========")
    results = evaluate_metrics(
        model=model,
        loader=loader,
        device=device,
        generator=generator,
        eps=cfg.eps,
        save_per_sample=cfg.save_per_sample,
        save_predictions=cfg.save_predictions,
    )

    total_params, trainable_params = count_parameters(model)
    param_mem_mb = param_memory_mb(total_params)
    ckpt_size_mb = get_ckpt_size_mb(cfg.ckpt_path)

    print()
    print("========== FLOPs ==========")
    flops = compute_flops(
        model=model,
        device=device,
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
    )

    if flops is not None:
        theory_time = theoretical_time_from_flops(
            flops=flops,
            gpu_tflops=cfg.gpu_tflops_fp32,
        )
    else:
        theory_time = float("nan")

    print()
    print("========== Test Results ==========")
    print(f"[NMSE]   test global: {results['global_nmse']:.6e} ({results['global_nmse_db']:.2f} dB)")
    print(f"[R2]     test global: {results['global_r2']:.6f}")
    print(f"[EVM]    test global: {results['global_evm_percent']:.4f}%")
    print(f"[Params] total: {total_params} ({format_count(total_params)})")
    print(f"[Params] trainable: {trainable_params} ({format_count(trainable_params)})")
    print(f"[Memory] parameter memory(fp32): {param_mem_mb:.2f} MB")
    print(f"[Size]   checkpoint: {ckpt_size_mb:.2f} MB")

    if flops is not None:
        print(f"[FLOPs]  single-sample forward: {flops:.0f} ({format_count(flops)})")
        print(f"[Time]   theoretical on {cfg.gpu_name_for_theory}: {theory_time:.6e} s")

    if cfg.save_per_sample:
        valid_nmse = results["nmse_per_sample"][~np.isnan(results["nmse_per_sample"])]
        valid_r2 = results["r2_per_sample"][~np.isnan(results["r2_per_sample"])]
        valid_evm = results["evm_per_sample"][~np.isnan(results["evm_per_sample"])]

        if valid_nmse.size > 0:
            print(
                f"[NMSE]   per-sample mean={valid_nmse.mean():.6e}, "
                f"std={valid_nmse.std():.6e}"
            )

        if valid_r2.size > 0:
            print(
                f"[R2]     per-sample mean={valid_r2.mean():.6f}, "
                f"std={valid_r2.std():.6f}, "
                f"NaN count={results['r2_per_sample'].size - valid_r2.size}"
            )

        if valid_evm.size > 0:
            print(
                f"[EVM]    per-sample mean={valid_evm.mean():.4f}%, "
                f"std={valid_evm.std():.4f}%"
            )

    save_results(
        cfg=cfg,
        results=results,
        total_params=total_params,
        trainable_params=trainable_params,
        param_mem_mb=param_mem_mb,
        ckpt_size_mb=ckpt_size_mb,
        flops=flops,
        theory_time=theory_time,
        device=device,
    )


if __name__ == "__main__":
    main()