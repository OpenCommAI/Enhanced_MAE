import os
import csv
import math
import random
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_mae_multi_hop_random_v1 import build_mae_channel_target_hop3


@dataclass
class Config:
    # ========================================================
    # 1. 数据路径
    # ========================================================
    # 数据总目录：
    # 例如:
    # ./Prediction_length_CSI_data/25dB_zscore_norm_8toX_svd090_dataset/
    # num1 = 8
    # data_root: str = fr"./Prediction_length_CSI_data/25dB_zscore_norm_8toX_svd090_dataset/"
    # all_dataset_name: str = fr"8to{num1}_interval1_gap1_seqsplit"

    num2 = 25
    num1 = 90

    data_root: str = fr"./data/spectral_efficiency/"

    all_dataset_name: str = fr"CSI_data_test_label"

    # 预测几个 TTI 就改这里：
    # 8to1 -> pred_tti = 1
    # 8to2 -> pred_tti = 2
    # 8to8 -> pred_tti = 8
    pred_tti: int = 1

    # 数据集子目录：
    # 如果 pred_tti=8，则默认读取:
    # 8to8_interval1_gap1_seqsplit/

    # 输出目录
    out_root: str = fr"./CSI_data_test_label/outputs_enhanced_mae_{num2}dB_svd0{num1}_8to1"

    # ========================================================
    # 2. 数据参数
    # ========================================================
    history_tti: int = 8

    # ========================================================
    # 3. 训练参数
    # ========================================================
    batch_size: int = 128
    num_workers: int = 1
    epochs: int = 300

    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    amp: bool = True
    seed: int = 42
    save_every: int = 100

    warmup_epochs: int = 10
    eta_min: float = 1e-5

    device: str = "cuda"

    # ========================================================
    # 4. 跳频参数
    # ========================================================
    hop_steps: Tuple[int, ...] = (1, 3, 5, 7)
    balanced_hop: bool = True

    # 验证时是否对所有 hop 求平均
    eval_all_hops: bool = True

    def __post_init__(self):
        if self.all_dataset_name == "":
            self.all_dataset_name = f"8to{self.pred_tti}_interval1_gap1_seqsplit"


cfg = Config()


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
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=amp_enabled)
    except Exception:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def autocast_ctx():
        if not amp_enabled:
            return nullcontext()

        try:
            from torch.cuda.amp import autocast
            return autocast()
        except Exception:
            return torch.amp.autocast(device_type="cuda")

    return scaler, autocast_ctx


def nmse_to_db(nmse_value: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(nmse_value), eps))


def format_hop_counts(hop_counts):
    return ", ".join([f"{h}:{hop_counts.get(h, 0)}" for h in cfg.hop_steps])


def complex_to_2ch(x: np.ndarray):
    assert x.ndim == 3, f"expect 3D complex array, got {x.ndim}D, shape={x.shape}"

    real = np.real(x)
    imag = np.imag(x)

    x = np.stack([real, imag], axis=1)

    return x.astype(np.float32)


def array_to_2ch(x: np.ndarray, name: str):
    """
    统一转成:
        (N, 2, 64, W)

    支持:
        complex 3D: (N, 64, W)
        real 3D   : (N, 64, W)
        4D        : (N, 2, 64, W)
        4D        : (N, 64, W, 2)
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
    X = np.load(inputs_path)
    Y = np.load(labels_path)

    print(f"Raw inputs : {X.shape}, {X.dtype}")
    print(f"Raw labels : {Y.shape}, {Y.dtype}")

    X = array_to_2ch(X, name=os.path.basename(inputs_path))
    Y = array_to_2ch(Y, name=os.path.basename(labels_path))

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    print(f"Processed inputs : {X.shape}, {X.dtype}")
    print(f"Processed labels : {Y.shape}, {Y.dtype}")

    return X, Y


def load_all_dataset(data_root: str):
    all_dir = os.path.join(data_root, cfg.all_dataset_name)

    train_inputs_path = os.path.join(all_dir, "train_inputs.npy")
    train_labels_path = os.path.join(all_dir, "train_labels.npy")
    test_inputs_path = os.path.join(all_dir, "test_inputs.npy")
    test_labels_path = os.path.join(all_dir, "test_labels.npy")

    for path in [
        train_inputs_path,
        train_labels_path,
        test_inputs_path,
        test_labels_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到数据文件: {path}")

    print("\n" + "=" * 100)
    print("加载总训练集和总测试集")
    print(f"All dataset dir: {all_dir}")
    print("=" * 100)

    Xtr, Ytr = load_pair(train_inputs_path, train_labels_path)
    Xte, Yte = load_pair(test_inputs_path, test_labels_path)

    return Xtr, Ytr, Xte, Yte


class TTIDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, history_tti: int, pred_tti: int):
        self.X = X
        self.Y = Y

        expected_x_shape = (2, 64, 32 * history_tti)
        expected_y_shape = (2, 64, 32 * pred_tti)

        assert self.X.shape[1:] == expected_x_shape, (
            f"X shape wrong: got {self.X.shape[1:]}, expected {expected_x_shape}"
        )

        assert self.Y.shape[1:] == expected_y_shape, (
            f"Y shape wrong: got {self.Y.shape[1:]}, expected {expected_y_shape}"
        )

        assert len(self.X) == len(self.Y), (
            f"X 和 Y 样本数量不一致: len(X)={len(self.X)}, len(Y)={len(self.Y)}"
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_loader(X, Y, use_cuda: bool, shuffle: bool):
    dataset = TTIDataset(
        X,
        Y,
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
    )

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


def build_model():
    model = build_mae_channel_target_hop3(
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
        hop_steps=cfg.hop_steps,
        balanced_hop=cfg.balanced_hop,
    )
    return model


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled=True):
    model.eval()

    use_cuda = device.type == "cuda"
    _, autocast_ctx = make_amp_things(use_cuda, amp_enabled)

    loss_sum = 0.0
    n = 0

    eval_hops = cfg.hop_steps if cfg.eval_all_hops else None

    for x_input, y_future in loader:
        xb = x_input.to(device, non_blocking=use_cuda)
        yb = y_future.to(device, non_blocking=use_cuda)

        with autocast_ctx():
            if eval_hops is None:
                loss, pred, mask, loss_dict = model(xb, yb)
                batch_loss = loss.item()
            else:
                batch_loss = 0.0

                for hop in eval_hops:
                    loss, pred, mask, loss_dict = model(
                        xb,
                        yb,
                        force_hop_step=int(hop),
                    )
                    batch_loss += loss.item()

                batch_loss /= len(eval_hops)

        bs = xb.size(0)
        loss_sum += batch_loss * bs
        n += bs

    nmse_mean = loss_sum / max(n, 1)
    nmse_db = nmse_to_db(nmse_mean)

    return nmse_mean, nmse_db


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    train_nmse_db,
    test_loss,
    test_nmse_db,
    best_epoch,
    best_test_loss,
    best_test_nmse_db,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,

            "epoch": epoch,

            "train_nmse": train_loss,
            "train_nmse_db": train_nmse_db,

            "test_nmse": test_loss,
            "test_nmse_db": test_nmse_db,

            "best_epoch": best_epoch,
            "best_test_nmse": best_test_loss,
            "best_test_nmse_db": best_test_nmse_db,

            "history_tti": cfg.history_tti,
            "pred_tti": cfg.pred_tti,
            "hop_steps": cfg.hop_steps,
            "balanced_hop": cfg.balanced_hop,
            "eval_all_hops": cfg.eval_all_hops,

            "config": asdict(cfg),
        },
        path,
    )


def main():
    assert cfg.history_tti == 8, "当前实验保持 history_tti=8"
    assert cfg.pred_tti >= 1, "pred_tti 必须 >= 1"
    assert cfg.epochs > cfg.warmup_epochs, "epochs 必须大于 warmup_epochs"

    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    use_cuda = device.type == "cuda"

    run_name = f"enhanced_mae_8to{cfg.pred_tti}_{timestamp()}"
    run_dir = os.path.join(cfg.out_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    log_path = os.path.join(run_dir, "training.log")
    csv_path = os.path.join(run_dir, "train_history.csv")

    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(str(asdict(cfg)) + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_nmse",
            "train_nmse_db",
            "test_nmse",
            "test_nmse_db",
            "lr",
            "train_hop_counts",
            "best_epoch",
            "best_test_nmse",
            "best_test_nmse_db",
        ])

    Xtr, Ytr, Xte, Yte = load_all_dataset(cfg.data_root)

    train_loader, test_loader, train_size, test_size = make_train_test_loaders(
        Xtr,
        Ytr,
        Xte,
        Yte,
        use_cuda=use_cuda,
    )

    model = build_model().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 100)
    print("Model type: enhanced_mae_multi_hop_8toX")
    print(f"Model Parameters: total={total_params:,}, trainable={trainable_params:,}")
    print(f"Device: {device}")
    print(f"Train samples: {train_size}")
    print(f"Total test samples: {test_size}")
    print(
        f"history_tti={cfg.history_tti}, "
        f"pred_tti={cfg.pred_tti}, "
        f"hop_steps={cfg.hop_steps}, "
        f"balanced_hop={cfg.balanced_hop}, "
        f"eval_all_hops={cfg.eval_all_hops}"
    )
    print(f"Config: {asdict(cfg)}")
    print("=" * 100)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("Model type: enhanced_mae_multi_hop_8toX\n")
        f.write(f"Model Parameters: total={total_params}, trainable={trainable_params}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Train samples: {train_size}\n")
        f.write(f"Total test samples: {test_size}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

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

    best_test_loss = float("inf")
    best_test_nmse_db = None
    best_epoch = 0

    last_train_loss = None
    last_train_nmse_db = None
    last_test_loss = None
    last_test_nmse_db = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        loss_running = 0.0
        seen = 0
        hop_counts = {int(h): 0 for h in cfg.hop_steps}

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.epochs}",
            ncols=120,
        )

        for x_input, y_future in pbar:
            xb = x_input.to(device, non_blocking=use_cuda)
            yb = y_future.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                loss, pred, mask, loss_dict = model(xb, yb)

            scaler.scale(loss).backward()

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
            loss_running += loss.item() * bs
            seen += bs

            train_loss_now = loss_running / max(seen, 1)
            train_nmse_db_now = nmse_to_db(train_loss_now)

            pbar.set_postfix(
                train_nmse=f"{train_loss_now:.3e}",
                train_db=f"{train_nmse_db_now:.2f}",
                hops=format_hop_counts(hop_counts),
            )

        train_loss = loss_running / max(seen, 1)
        train_nmse_db = nmse_to_db(train_loss)

        total_test_loss, total_test_nmse_db = evaluate(
            model,
            test_loader,
            device,
            amp_enabled=cfg.amp,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        last_train_loss = train_loss
        last_train_nmse_db = train_nmse_db
        last_test_loss = total_test_loss
        last_test_nmse_db = total_test_nmse_db

        if total_test_loss < best_test_loss:
            best_test_loss = total_test_loss
            best_test_nmse_db = total_test_nmse_db
            best_epoch = epoch

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                train_nmse_db=train_nmse_db,
                test_loss=total_test_loss,
                test_nmse_db=total_test_nmse_db,
                best_epoch=best_epoch,
                best_test_loss=best_test_loss,
                best_test_nmse_db=best_test_nmse_db,
            )

        line = (
            f"epoch: {epoch}, "
            f"model_type: enhanced_mae_multi_hop_8toX, "
            f"train_nmse: {train_loss:.6e} ({train_nmse_db:.2f} dB), "
            f"total_test_nmse: {total_test_loss:.6e} ({total_test_nmse_db:.2f} dB), "
            f"lr: {lr_now:.6g}, "
            f"train_hop_counts: {format_hop_counts(hop_counts)}, "
            f"best_epoch: {best_epoch}, "
            f"best_total_test_nmse: {best_test_loss:.6e} "
            f"({best_test_nmse_db:.2f} dB)"
        )

        print(line)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                train_nmse_db,
                total_test_loss,
                total_test_nmse_db,
                lr_now,
                format_hop_counts(hop_counts),
                best_epoch,
                best_test_loss,
                best_test_nmse_db,
            ])

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                path=os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                train_nmse_db=train_nmse_db,
                test_loss=total_test_loss,
                test_nmse_db=total_test_nmse_db,
                best_epoch=best_epoch,
                best_test_loss=best_test_loss,
                best_test_nmse_db=best_test_nmse_db,
            )

    save_checkpoint(
        path=last_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=cfg.epochs,
        train_loss=last_train_loss,
        train_nmse_db=last_train_nmse_db,
        test_loss=last_test_loss,
        test_nmse_db=last_test_nmse_db,
        best_epoch=best_epoch,
        best_test_loss=best_test_loss,
        best_test_nmse_db=best_test_nmse_db,
    )

    print("\n" + "=" * 100)
    print("训练完成")
    print(f"输出目录: {run_dir}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best loss / NMSE: {best_test_loss:.6e}")
    print(f"Best NMSE(dB): {best_test_nmse_db:.2f} dB")
    print(f"Best model path: {best_path}")
    print(f"Last model path: {last_path}")
    print("=" * 100)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\nTraining finished.\n")
        f.write(f"Output dir: {run_dir}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best loss / NMSE: {best_test_loss:.6e}\n")
        f.write(f"Best NMSE(dB): {best_test_nmse_db:.2f} dB\n")
        f.write(f"Best model path: {best_path}\n")
        f.write(f"Last model path: {last_path}\n")


if __name__ == "__main__":
    main()