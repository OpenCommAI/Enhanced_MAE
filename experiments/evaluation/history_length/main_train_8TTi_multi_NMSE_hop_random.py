import os
import csv
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from models_mae_multi_hop_random_v1 import build_mae_channel_target_hop3


# ============================================================
# 1. 配置
# ============================================================
@dataclass
class Config:
    # ========================================================
    # 数据路径
    # 这个目录下应该直接包含:
    #   train_inputs.npy
    #   train_labels.npy
    #   test_inputs.npy
    #   test_labels.npy
    # ========================================================
    history_tti: int = 8

    data_dir: str = fr"./CSI_data_25dB_test/CSI_data_v1_0602_v1/Xtti_to_1tti_dataset_svd090_norm/{history_tti}to1"

    # 输出目录
    out_root: str = fr"./result_0602/mae_realimag_loss_X{history_tti}"

    # ========================================================
    # 数据参数
    # 你当前 train_inputs shape = (10580, 64, 320)
    # 所以 history_tti = 10
    # ========================================================
    pred_tti: int = 1

    # ========================================================
    # 训练参数
    # ========================================================
    batch_size: int = 128
    num_workers: int = 1
    epochs: int = 300

    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    amp: bool = True
    seed: int = 42
    save_every: int = 50

    warmup_epochs: int = 10
    eta_min: float = 1e-6

    device: str = "cuda"

    # ========================================================
    # 跳频参数
    # ========================================================
    hop_steps: Tuple[int, ...] = (1, 3, 5, 7)
    balanced_hop: bool = True

    # 测试时分别用 1、3、5、7 做 forward，然后对 loss 求平均
    eval_all_hops: bool = True

    # 训练损失选择:
    #   "total": 实部+虚部总 NMSE
    #   "real" : 只训练实部 NMSE
    #   "imag" : 只训练虚部 NMSE
    #   "mean_real_imag": 实部 NMSE 和虚部 NMSE 的平均
    train_loss_mode: str = "total"


cfg = Config()


# ============================================================
# 2. 工具函数
# ============================================================
def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def make_amp_things(use_cuda: bool, amp_enabled: bool):
    amp_enabled = bool(use_cuda and amp_enabled)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        try:
            scaler = torch.amp.GradScaler(enabled=amp_enabled)
        except Exception:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler(enabled=amp_enabled)

    def autocast_ctx():
        if not amp_enabled:
            return nullcontext()

        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast(enabled=True)

    return scaler, autocast_ctx


def nmse_to_db(nmse_value: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(nmse_value), eps))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def format_hop_counts(hop_counts):
    return ", ".join([f"{h}:{hop_counts.get(h, 0)}" for h in cfg.hop_steps])


# ============================================================
# 3. 数据读取
# ============================================================
def complex_to_2ch(x: np.ndarray):
    """
    输入:
        complex x: (N, 64, 32*k)

    输出:
        float32 x: (N, 2, 64, 32*k)

    通道 0: 实部
    通道 1: 虚部
    """
    assert x.ndim == 3, f"expect 3D complex array, got {x.ndim}D, shape={x.shape}"

    real = np.real(x)
    imag = np.imag(x)

    x = np.stack(
        [
            real,
            imag,
        ],
        axis=1,
    )

    return x.astype(np.float32)


def array_to_2ch(x: np.ndarray, name: str):
    """
    支持:
        1. complex 3D: (N, 64, W)
        2. real 3D   : (N, 64, W)，虚部补 0
        3. 4D        : (N, 2, 64, W)
        4. 4D        : (N, 64, W, 2)
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


def load_pair(inputs_path: str, labels_path: str):
    if not os.path.exists(inputs_path):
        raise FileNotFoundError(f"找不到输入文件: {inputs_path}")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"找不到标签文件: {labels_path}")

    X = np.load(inputs_path)
    Y = np.load(labels_path)

    print(f"Raw inputs : {X.shape}, {X.dtype}")
    print(f"Raw labels : {Y.shape}, {Y.dtype}")

    X = array_to_2ch(X, name=os.path.basename(inputs_path))
    Y = array_to_2ch(Y, name=os.path.basename(labels_path))

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    expected_x_shape = (2, 64, 32 * cfg.history_tti)
    expected_y_shape = (2, 64, 32 * cfg.pred_tti)

    assert X.shape[1:] == expected_x_shape, (
        f"X shape wrong: got {X.shape[1:]}, expected {expected_x_shape}"
    )

    assert Y.shape[1:] == expected_y_shape, (
        f"Y shape wrong: got {Y.shape[1:]}, expected {expected_y_shape}"
    )

    assert len(X) == len(Y), (
        f"X 和 Y 样本数量不一致: len(X)={len(X)}, len(Y)={len(Y)}"
    )

    print(f"Processed inputs : {X.shape}, {X.dtype}")
    print(f"Processed labels : {Y.shape}, {Y.dtype}")

    return X, Y


def load_dataset(data_dir: str):
    train_inputs_path = os.path.join(data_dir, "train_inputs.npy")
    train_labels_path = os.path.join(data_dir, "train_labels.npy")
    test_inputs_path = os.path.join(data_dir, "test_inputs.npy")
    test_labels_path = os.path.join(data_dir, "test_labels.npy")

    print("\n" + "=" * 100)
    print("加载 complex XTTI -> 1TTI 数据集")
    print(f"Data dir: {data_dir}")
    print("=" * 100)

    Xtr, Ytr = load_pair(train_inputs_path, train_labels_path)
    Xte, Yte = load_pair(test_inputs_path, test_labels_path)

    return Xtr, Ytr, Xte, Yte


class TTIDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert len(self.X) == len(self.Y), (
            f"X 和 Y 样本数量不一致: len(X)={len(self.X)}, len(Y)={len(self.Y)}"
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_loader(X, Y, use_cuda: bool, shuffle: bool):
    dataset = TTIDataset(X, Y)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    return loader, len(dataset)


def make_train_test_loaders(Xtr, Ytr, Xte, Yte, use_cuda: bool):
    train_loader, train_size = make_loader(
        Xtr,
        Ytr,
        use_cuda=use_cuda,
        shuffle=True,
    )

    test_loader, test_size = make_loader(
        Xte,
        Yte,
        use_cuda=use_cuda,
        shuffle=False,
    )

    return train_loader, test_loader, train_size, test_size


# ============================================================
# 4. 模型构建
# ============================================================
def build_model():
    model = build_mae_channel_target_hop3(
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
        hop_steps=cfg.hop_steps,
        balanced_hop=cfg.balanced_hop,
    )

    return model


# ============================================================
# 5. loss 选择
# ============================================================
def select_train_loss(loss_dict):
    mode = cfg.train_loss_mode.lower()

    if mode == "total":
        return loss_dict["nmse_total"]

    if mode == "real":
        return loss_dict["nmse_real"]

    if mode == "imag":
        return loss_dict["nmse_imag"]

    if mode == "mean_real_imag":
        return 0.5 * (loss_dict["nmse_real"] + loss_dict["nmse_imag"])

    raise ValueError(
        f"Unsupported train_loss_mode={cfg.train_loss_mode}, "
        f"choose from total/real/imag/mean_real_imag"
    )


# ============================================================
# 6. 评估函数
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, amp_enabled=True):
    model.eval()

    use_cuda = device.type == "cuda"
    _, autocast_ctx = make_amp_things(use_cuda, amp_enabled)

    eval_hops = cfg.hop_steps if cfg.eval_all_hops else None

    total_sum = 0.0
    real_sum = 0.0
    imag_sum = 0.0
    n = 0

    for x_input, y_future in loader:
        xb = x_input.to(device, non_blocking=use_cuda)
        yb = y_future.to(device, non_blocking=use_cuda)

        bs = xb.size(0)

        with autocast_ctx():
            if eval_hops is None:
                loss, pred, mask, loss_dict = model(xb, yb)

                batch_total = loss_dict["nmse_total"].item()
                batch_real = loss_dict["nmse_real"].item()
                batch_imag = loss_dict["nmse_imag"].item()

            else:
                batch_total = 0.0
                batch_real = 0.0
                batch_imag = 0.0

                for hop in eval_hops:
                    loss, pred, mask, loss_dict = model(
                        xb,
                        yb,
                        force_hop_step=int(hop),
                    )

                    batch_total += loss_dict["nmse_total"].item()
                    batch_real += loss_dict["nmse_real"].item()
                    batch_imag += loss_dict["nmse_imag"].item()

                batch_total /= len(eval_hops)
                batch_real /= len(eval_hops)
                batch_imag /= len(eval_hops)

        total_sum += batch_total * bs
        real_sum += batch_real * bs
        imag_sum += batch_imag * bs
        n += bs

    mean_total = total_sum / max(n, 1)
    mean_real = real_sum / max(n, 1)
    mean_imag = imag_sum / max(n, 1)

    return {
        "nmse_total": mean_total,
        "nmse_total_db": nmse_to_db(mean_total),
        "nmse_real": mean_real,
        "nmse_real_db": nmse_to_db(mean_real),
        "nmse_imag": mean_imag,
        "nmse_imag_db": nmse_to_db(mean_imag),
    }


# ============================================================
# 7. checkpoint
# ============================================================
def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    train_metrics,
    test_metrics,
    best_epoch,
    best_test_total,
    best_test_total_db,
):
    ensure_dir(os.path.dirname(path))

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "best_epoch": best_epoch,
            "best_test_total": best_test_total,
            "best_test_total_db": best_test_total_db,
            "history_tti": cfg.history_tti,
            "pred_tti": cfg.pred_tti,
            "hop_steps": cfg.hop_steps,
            "balanced_hop": cfg.balanced_hop,
            "eval_all_hops": cfg.eval_all_hops,
            "train_loss_mode": cfg.train_loss_mode,
            "config": asdict(cfg),
        },
        path,
    )


# ============================================================
# 8. 主函数
# ============================================================
def main():
    assert cfg.history_tti >= 1, "history_tti must be >= 1"
    assert cfg.pred_tti == 1, "当前模型用于 X 个 TTI 预测 1 个 TTI"
    assert cfg.epochs > cfg.warmup_epochs

    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    use_cuda = device.type == "cuda"

    run_name = (
        f"mae_realimag_"
        f"X{cfg.history_tti}_to_1tti_"
        f"loss_{cfg.train_loss_mode}_"
        f"{timestamp()}"
    )

    run_dir = os.path.join(cfg.out_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    log_path = os.path.join(run_dir, "training.log")
    csv_path = os.path.join(run_dir, "metrics.csv")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(str(asdict(cfg)) + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_total_nmse",
                "train_total_db",
                "train_real_nmse",
                "train_real_db",
                "train_imag_nmse",
                "train_imag_db",
                "test_total_nmse",
                "test_total_db",
                "test_real_nmse",
                "test_real_db",
                "test_imag_nmse",
                "test_imag_db",
            ]
        )

    Xtr, Ytr, Xte, Yte = load_dataset(cfg.data_dir)

    train_loader, test_loader, train_size, test_size = make_train_test_loaders(
        Xtr,
        Ytr,
        Xte,
        Yte,
        use_cuda=use_cuda,
    )

    model = build_model().to(device)

    total_params, trainable_params = count_parameters(model)

    header = [
        "=" * 100,
        "Model type: MAE multi-hop real/imag loss experiment",
        f"Device: {device}",
        f"Train samples: {train_size}",
        f"Test samples : {test_size}",
        f"Input shape : (B, 2, 64, {32 * cfg.history_tti})",
        f"Output shape: (B, 2, 64, 32)",
        f"history_tti X: {cfg.history_tti}",
        f"train_loss_mode: {cfg.train_loss_mode}",
        f"hop_steps={cfg.hop_steps}, balanced_hop={cfg.balanced_hop}, eval_all_hops={cfg.eval_all_hops}",
        f"Model Parameters: total={total_params:,}, trainable={trainable_params:,}",
        f"Config: {asdict(cfg)}",
        "=" * 100,
    ]

    print("\n".join(header))

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    warmup = LinearLR(
        optimizer,
        start_factor=0.2,
        total_iters=cfg.warmup_epochs,
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.warmup_epochs,
        eta_min=cfg.eta_min,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.warmup_epochs],
    )

    scaler, autocast_ctx = make_amp_things(use_cuda, cfg.amp)

    best_test_total = float("inf")
    best_test_total_db = None
    best_epoch = 0

    last_train_metrics = None
    last_test_metrics = None

    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        train_total_sum = 0.0
        train_real_sum = 0.0
        train_imag_sum = 0.0
        seen = 0

        hop_counts = {int(h): 0 for h in cfg.hop_steps}

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.epochs}",
            ncols=140,
        )

        for x_input, y_future in pbar:
            xb = x_input.to(device, non_blocking=use_cuda)
            yb = y_future.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                loss_total, pred, mask, loss_dict = model(xb, yb)
                loss_for_backward = select_train_loss(loss_dict)

            scaler.scale(loss_for_backward).backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if "hop_steps" in loss_dict:
                hops = loss_dict["hop_steps"].detach().cpu().tolist()
                for h in hops:
                    h = int(h)
                    hop_counts[h] = hop_counts.get(h, 0) + 1

            bs = xb.size(0)

            train_total_sum += loss_dict["nmse_total"].item() * bs
            train_real_sum += loss_dict["nmse_real"].item() * bs
            train_imag_sum += loss_dict["nmse_imag"].item() * bs
            seen += bs

            train_total_now = train_total_sum / max(seen, 1)
            train_real_now = train_real_sum / max(seen, 1)
            train_imag_now = train_imag_sum / max(seen, 1)

            pbar.set_postfix(
                total_db=f"{nmse_to_db(train_total_now):.2f}",
                real_db=f"{nmse_to_db(train_real_now):.2f}",
                imag_db=f"{nmse_to_db(train_imag_now):.2f}",
                hops=format_hop_counts(hop_counts),
            )

        train_total = train_total_sum / max(seen, 1)
        train_real = train_real_sum / max(seen, 1)
        train_imag = train_imag_sum / max(seen, 1)

        train_metrics = {
            "nmse_total": train_total,
            "nmse_total_db": nmse_to_db(train_total),
            "nmse_real": train_real,
            "nmse_real_db": nmse_to_db(train_real),
            "nmse_imag": train_imag,
            "nmse_imag_db": nmse_to_db(train_imag),
        }

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            amp_enabled=cfg.amp,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        last_train_metrics = train_metrics
        last_test_metrics = test_metrics

        line = (
            f"epoch: {epoch}, "
            f"train_total: {train_metrics['nmse_total']:.6e} ({train_metrics['nmse_total_db']:.2f} dB), "
            f"train_real: {train_metrics['nmse_real']:.6e} ({train_metrics['nmse_real_db']:.2f} dB), "
            f"train_imag: {train_metrics['nmse_imag']:.6e} ({train_metrics['nmse_imag_db']:.2f} dB), "
            f"test_total: {test_metrics['nmse_total']:.6e} ({test_metrics['nmse_total_db']:.2f} dB), "
            f"test_real: {test_metrics['nmse_real']:.6e} ({test_metrics['nmse_real_db']:.2f} dB), "
            f"test_imag: {test_metrics['nmse_imag']:.6e} ({test_metrics['nmse_imag_db']:.2f} dB), "
            f"lr: {lr_now:.6g}, "
            f"train_hop_counts: {format_hop_counts(hop_counts)}"
        )

        print(line)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    lr_now,
                    train_metrics["nmse_total"],
                    train_metrics["nmse_total_db"],
                    train_metrics["nmse_real"],
                    train_metrics["nmse_real_db"],
                    train_metrics["nmse_imag"],
                    train_metrics["nmse_imag_db"],
                    test_metrics["nmse_total"],
                    test_metrics["nmse_total_db"],
                    test_metrics["nmse_real"],
                    test_metrics["nmse_real_db"],
                    test_metrics["nmse_imag"],
                    test_metrics["nmse_imag_db"],
                ]
            )

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                path=os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                best_epoch=best_epoch,
                best_test_total=best_test_total,
                best_test_total_db=best_test_total_db,
            )

        if test_metrics["nmse_total"] < best_test_total:
            best_test_total = test_metrics["nmse_total"]
            best_test_total_db = test_metrics["nmse_total_db"]
            best_epoch = epoch

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                best_epoch=best_epoch,
                best_test_total=best_test_total,
                best_test_total_db=best_test_total_db,
            )

            print(
                f"保存 best.pt: epoch={best_epoch}, "
                f"best_test_total={best_test_total:.6e}, "
                f"best_test_total_db={best_test_total_db:.2f} dB"
            )

    save_checkpoint(
        path=last_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=cfg.epochs,
        train_metrics=last_train_metrics,
        test_metrics=last_test_metrics,
        best_epoch=best_epoch,
        best_test_total=best_test_total,
        best_test_total_db=best_test_total_db,
    )

    print("\n训练完成。")
    print(f"输出目录: {run_dir}")
    print(f"metrics.csv: {csv_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best test total NMSE: {best_test_total:.6e}")
    print(f"Best test total NMSE dB: {best_test_total_db:.2f} dB")
    print(f"Best model path: {best_path}")
    print(f"Last model path: {last_path}")


if __name__ == "__main__":
    main()