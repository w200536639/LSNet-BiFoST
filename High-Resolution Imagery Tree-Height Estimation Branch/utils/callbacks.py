import os
import numpy as np
import torch
import scipy.signal
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory:
    """
    用于记录树高回归任务的训练与验证损失
    - 自动兼容浮点树高任务
    - 自动创建日志目录（exist_ok=True）
    - 保存 Loss 曲线与 TensorBoard 日志
    """
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag

        self.losses = []
        self.val_loss = [] if val_loss_flag else None

        # ✅ 目录已存在则跳过，避免 FileExistsError
        os.makedirs(self.log_dir, exist_ok=True)

        # TensorBoard writer 初始化
        self.writer = SummaryWriter(self.log_dir)

        # 尝试写入模型结构图（仅首次成功即可）
        try:
            dummy_input = torch.randn(1, 4, input_shape[0], input_shape[1])  # 4 通道输入
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"[LossHistory] ⚠️ 模型结构图写入失败：{e}")

    # --------------------------------------------------- #
    #   记录每个 epoch 的 loss
    # --------------------------------------------------- #
    def append_loss(self, epoch, loss, val_loss=None):
        # 防御性检测目录
        os.makedirs(self.log_dir, exist_ok=True)

        self.losses.append(loss)
        if self.val_loss_flag and val_loss is not None:
            self.val_loss.append(val_loss)

        # 写入文本日志
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), "a", encoding="utf-8") as f:
            f.write(f"{loss:.6f}\n")
        if self.val_loss_flag and val_loss is not None:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), "a", encoding="utf-8") as f:
                f.write(f"{val_loss:.6f}\n")

        # 写入 TensorBoard
        self.writer.add_scalar("train/loss", loss, epoch)
        if self.val_loss_flag and val_loss is not None:
            self.writer.add_scalar("val/loss", val_loss, epoch)

        # 绘制 Loss 曲线
        self.loss_plot()

    # --------------------------------------------------- #
    #   绘制 Loss 曲线
    # --------------------------------------------------- #
    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()

        plt.plot(iters, self.losses, "r", linewidth=2, label="Train Loss")
        if self.val_loss_flag and len(self.val_loss) > 0:
            plt.plot(iters, self.val_loss, "orange", linewidth=2, label="Val Loss")

        # 平滑曲线（Savitzky-Golay 滤波）
        try:
            if len(self.losses) >= 7:
                window = min(len(self.losses) // 2 * 2 + 1, 15)
                plt.plot(iters, scipy.signal.savgol_filter(self.losses, window, 3),
                         "g--", linewidth=2, label="Smooth Train Loss")
                if self.val_loss_flag and len(self.val_loss) >= 7:
                    plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, window, 3),
                             "b--", linewidth=2, label="Smooth Val Loss")
        except Exception as e:
            print(f"[LossHistory] ⚠️ 平滑曲线绘制失败：{e}")

        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"), dpi=200)
        plt.close()

    # --------------------------------------------------- #
    #   关闭 TensorBoard writer
    # --------------------------------------------------- #
    def close(self):
        self.writer.close()
