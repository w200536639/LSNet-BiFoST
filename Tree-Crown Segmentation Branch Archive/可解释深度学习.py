import os
import gc
from functools import partial

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ====== 让 Matplotlib 支持中文 / 全角括号，避免 glyph missing ======
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ===== Nature 子刊风格的图像风格 =====
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
})

# ====== 工程内导入，保持和训练脚本一致 ======
from nets.lsnet_bifost_segmentation import Unet
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, worker_init_fn


# ============================================================
# 全局配置（按你 LSNet 树冠分割实验修改）
# ============================================================
CUDA            = True
SEED            = 11

# 和训练脚本保持一致
NUM_CLASSES     = 2                 # 背景 + 梭梭
INPUT_SHAPE     = (640, 640)        # 和训练相同
BACKBONE        = "lsnet_b"         # lsnet_t / lsnet_s / lsnet_b
USE_BIE         = True              # 对应 use_bie=1
USE_HPA         = True              # 对应 use_hpa=1

# 模型权重（改成你的 best_epoch_weights.pth）
MODEL_PATH      = r"model_data\best_epoch_weights.pth"

# 数据集根目录（和训练脚本 VOC_PATH 一致）
VOC_PATH        = r"VOCdevkit"
VAL_TXT         = os.path.join(VOC_PATH, "VOC2007", "ImageSets", "Segmentation", "val.txt")

BATCH_SIZE      = 1                 # Grad-CAM 建议 1
NUM_WORKERS     = 0
TREE_CLASS_IDX  = 1                 # 梭梭类别通道索引（0 背景 / 1 梭梭）
MAX_IMAGES      = 236               # 最多可视化多少张验证图

# 输出目录 & 标题
SAVE_ROOT       = "gradcam_lsnet_bifost"
PAPER_TAG       = "LSNet-BiFoST tree-crown segmentation Grad-CAM"


# ============================================================
# Grad-CAM 核心类
# ============================================================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, layer_name: str):
        self.model = model
        self.target_layer = target_layer
        self.layer_name = layer_name
        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            # 保存前向特征图
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # 保存反向梯度
            self.gradients = grad_out[0].detach()

        self.fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

        # 方便外部调试
        self._last_score = None

    def generate(self, input_tensor: torch.Tensor, class_idx: int):
        """
        input_tensor: [1, C, H, W]
        class_idx:    目标类别索引（1=梭梭）
        """
        assert input_tensor.dim() == 4 and input_tensor.size(0) == 1

        # 清梯度
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)

        # 分割任务：对该类别通道做全局平均作为“score”
        if output.dim() == 4:
            score = output[:, class_idx, :, :].mean()
        else:
            score = output[:, class_idx].mean()

        self._last_score = score.item()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("未捕获到激活或梯度，请检查目标层是否正确注册。")

        # GAP 得到每个通道的权重
        # activations: [1, C, H, W]；gradients: [1, C, H, W]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1)             # [1, H, W]
        cam = F.relu(cam)[0]                                      # [H, W]

        cam_min, cam_max = cam.min(), cam.max()
        cam = cam - cam_min
        if (cam_max - cam_min) > 1e-6:
            cam = cam / (cam_max - cam_min)
        cam = cam.clamp(0.0, 1.0)

        return cam.detach().cpu().numpy().astype(np.float32)

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


# ============================================================
# 辅助函数
# ============================================================
def tensor_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """
    将张量还原为 0~255 的 RGB 图用于可视化。
    假设 img_tensor 形状为 [C,H,W]，范围约在 0~1。
    """
    img = img_tensor.detach().cpu().numpy()
    if img.shape[0] > 3:
        img = img[:3]     # 只取前三个通道
    img = np.transpose(img, (1, 2, 0))  # [H,W,C]
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    return img


def visualize_cam_nature(cam: np.ndarray, rgb_img: np.ndarray, alpha: float = 0.55):
    """
    Nature 子刊风格的叠加：
      - 使用感知更线性的 'magma' 色图
      - 背景保持原色，前景高响应区轻微高亮
    """
    H, W, _ = rgb_img.shape

    cam_norm = np.clip(cam, 0.0, 1.0)
    cam_uint8 = (cam_norm * 255).astype(np.uint8)
    cam_resized = np.array(
        Image.fromarray(cam_uint8).resize((W, H), Image.BILINEAR)
    )
    cam_resized = cam_resized.astype(np.float32) / 255.0

    cmap = plt.get_cmap("magma")
    heatmap = cmap(cam_resized)[..., :3]  # [H,W,3] in 0~1
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = (1 - alpha) * rgb_img.astype(np.float32) + alpha * heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return heatmap, overlay


def get_decoder_target_layers(model: nn.Module):
    """
    针对当前的 LSNet + BIE + HPA U-Net：
      重点关注 1/16,1/8,1/4,1/2 的解码阶段 + 输出头.
      即: up4 / up3 / up2 / up1 / out_head / final
    """
    layers = []
    for attr_name in ["up4", "up3", "up2", "up1"]:
        if hasattr(model, attr_name):
            blk = getattr(model, attr_name)
            if hasattr(blk, "conv"):
                layers.append((blk.conv, f"{attr_name}_conv"))

    if hasattr(model, "out_head"):
        layers.append((model.out_head, "out_head"))

    if hasattr(model, "final"):
        layers.append((model.final, "final_logits"))

    if not layers:
        raise ValueError("未在模型中找到 up4/up3/up2/up1/out_head/final 这些解码层，请检查网络定义。")
    return layers


def load_val_list(val_path: str):
    """
    兼容 utf-8 / gbk 的 val.txt 读取，避免中文文件名乱码。
    每行一般是图像 ID：如 2023071707（可见光）_row_10_col_13
    """
    for enc in ("utf-8", "gbk"):
        try:
            with open(val_path, "r", encoding=enc) as f:
                lines = [x.strip() for x in f if x.strip()]
            print(f"val.txt 使用编码: {enc}")
            return lines
        except UnicodeDecodeError:
            continue

    # 兜底方案
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [x.strip() for x in f if x.strip()]
    print("警告: 无法以 utf-8/gbk 完全解码 val.txt，已使用 utf-8+ignore，可能出现文件名乱码。")
    return lines


# ============================================================
# 主流程：对验证集跑 Grad-CAM
# ============================================================
def run_gradcam_on_valset():
    seed_everything(SEED)

    device = torch.device("cuda" if (CUDA and torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")

    # ---------- 构建模型（结构必须与训练时保持一致） ----------
    model = Unet(
        num_classes = NUM_CLASSES,
        pretrained  = False,        # LSNet 不依赖外部预训练
        backbone    = BACKBONE,
        use_bie     = USE_BIE,
        use_hpa     = USE_HPA,
    )

    # 取消所有 inplace ReLU，避免 Grad-CAM 反向时梯度被覆盖
    for m in model.modules():
        if hasattr(m, "inplace") and m.inplace:
            m.inplace = False

    # ---------- 加载训练好的权重 ----------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到模型权重: {MODEL_PATH}")

    print(f"加载模型权重: {MODEL_PATH}")
    try:
        # 新版 PyTorch 推荐 weights_only=True
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        # 兼容旧版 PyTorch（没有 weights_only 参数）
        state_dict = torch.load(MODEL_PATH, map_location=device)

    # 兼容 'state_dict' / DataParallel 前缀
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    clean_state = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        clean_state[new_k] = v

    model_dict = model.state_dict()
    load_keys, skip_keys = [], []
    for k, v in clean_state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            load_keys.append(k)
        else:
            skip_keys.append(k)
    model.load_state_dict(model_dict, strict=False)
    print(f"  成功加载 {len(load_keys)} 个权重，跳过 {len(skip_keys)} 个。")

    model.to(device)
    model.eval()

    # ---------- 获取解码器中的目标层 ----------
    target_layers = get_decoder_target_layers(model)
    print("\n将对以下解码器层做 Grad-CAM：")
    for _, name in target_layers:
        print("  -", name)

    # ---------- 构建验证集 DataLoader（与训练同一套 UnetDataset） ----------
    if not os.path.exists(VAL_TXT):
        raise FileNotFoundError(f"未找到验证集列表: {VAL_TXT}")

    val_lines = load_val_list(VAL_TXT)
    print(f"\n验证样本数: {len(val_lines)}")

    val_dataset = UnetDataset(
        annotation_lines = val_lines,
        input_shape      = INPUT_SHAPE,
        num_classes      = NUM_CLASSES,
        train            = False,
        dataset_path     = VOC_PATH,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size      = BATCH_SIZE,
        shuffle         = False,
        num_workers     = NUM_WORKERS,
        pin_memory      = True,
        drop_last       = False,
        collate_fn      = unet_dataset_collate,
        worker_init_fn  = partial(worker_init_fn, rank=0, seed=SEED),
    )

    # ---------- 输出目录 ----------
    os.makedirs(SAVE_ROOT, exist_ok=True)
    # 三联图目录
    triplet_dir = os.path.join(SAVE_ROOT, "triplet")
    os.makedirs(triplet_dir, exist_ok=True)

    img_counter = 0
    line_idx    = 0   # 用于在 val_lines 中索引当前图像 ID

    print("\n开始对验证集进行 Grad-CAM 可解释性分析...")
    for step, batch in enumerate(tqdm(val_loader, desc="Grad-CAM on val")):
        # -------- 通用拆包：只拿第一个元素作为图像 --------
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        # --------------------------------------------------
        bsz = images.size(0)

        for b in range(bsz):
            if img_counter >= MAX_IMAGES or line_idx >= len(val_lines):
                break

            # 按顺序从 val_lines 取对应 ID
            line   = val_lines[line_idx]
            img_id = line.split()[0]
            line_idx += 1

            # 为避免全角括号，转换成半角括号；中文保留
            safe_img_id = img_id.replace("（", "(").replace("）", ")")

            img_counter += 1

            # [1,C,H,W]
            input_tensor = images[b:b+1].to(device, dtype=torch.float32)
            rgb_vis = tensor_to_rgb(images[b])

            # 先跑一次前向，得到平均 p(tree)，用于标题展示
            with torch.no_grad():
                logits = model(input_tensor)
                prob = torch.softmax(logits, dim=1)[0]
                tree_prob = prob[TREE_CLASS_IDX].mean().item()

            # 对每个解码层做 Grad-CAM
            for layer, layer_name in target_layers:
                cam_engine = GradCAM(model, layer, layer_name)
                cam_tree = cam_engine.generate(input_tensor, TREE_CLASS_IDX)
                heatmap, overlay = visualize_cam_nature(cam_tree, rgb_vis, alpha=0.55)

                # ==== 为该层准备单独目录：/layer_name/rgb, /cam, /overlay ====
                layer_root   = os.path.join(SAVE_ROOT, layer_name)
                rgb_dir      = os.path.join(layer_root, "rgb")
                cam_dir      = os.path.join(layer_root, "cam")
                overlay_dir  = os.path.join(layer_root, "overlay")
                for d in [layer_root, rgb_dir, cam_dir, overlay_dir]:
                    os.makedirs(d, exist_ok=True)

                # 基础文件名：index + 图像 ID + 层名
                base_name = f"val_{img_counter:04d}_{safe_img_id}_{layer_name}"

                # ======================================================
                # 1. 三联图：RGB / Grad-CAM / Overlay （Nature 风格排版）
                # ======================================================
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))

                axes[0].imshow(rgb_vis)
                axes[0].set_title("Input RGB", fontsize=8)
                axes[0].axis("off")

                im1 = axes[1].imshow(cam_tree, cmap="magma", vmin=0.0, vmax=1.0)
                axes[1].set_title(f"Grad-CAM (tree crown)\n[{layer_name}]", fontsize=8)
                axes[1].axis("off")
                cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)

                axes[2].imshow(overlay)
                axes[2].set_title(f"Overlay\nMean p(tree)={tree_prob:.2f}", fontsize=8)
                axes[2].axis("off")

                fig.suptitle(
                    PAPER_TAG,
                    fontdict={"family": "Times New Roman", "size": 11},
                    y=1.02
                )
                plt.tight_layout()

                triplet_path = os.path.join(triplet_dir, base_name + "_triplet.png")
                fig.savefig(triplet_path, bbox_inches="tight")
                plt.close(fig)

                # ======================================================
                # 2. 单图输出：仅 RGB
                # ======================================================
                fig_rgb, ax_rgb = plt.subplots(1, 1, figsize=(3, 3))
                ax_rgb.imshow(rgb_vis)
                ax_rgb.axis("off")
                fig_rgb.suptitle(
                    PAPER_TAG,
                    fontdict={"family": "Times New Roman", "size": 11}
                )
                plt.tight_layout()
                rgb_path = os.path.join(rgb_dir, base_name + "_rgb.png")
                fig_rgb.savefig(rgb_path, bbox_inches="tight")
                plt.close(fig_rgb)

                # ======================================================
                # 3. 单图输出：仅 Grad-CAM（带 colorbar，但不要任何文字标签）
                #    不设置 title / suptitle，只保留色带
                # ======================================================
                fig_cam, ax_cam = plt.subplots(1, 1, figsize=(3, 3))
                im_cam = ax_cam.imshow(cam_tree, cmap="magma", vmin=0.0, vmax=1.0)
                ax_cam.axis("off")
                fig_cam.colorbar(im_cam, fraction=0.046, pad=0.04)
                # 不调用 fig_cam.suptitle()，不写任何文字
                plt.tight_layout()
                cam_path = os.path.join(cam_dir, base_name + "_cam.png")
                fig_cam.savefig(cam_path, bbox_inches="tight")
                plt.close(fig_cam)

                # ======================================================
                # 4. 单图输出：仅 Overlay
                # ======================================================
                fig_ov, ax_ov = plt.subplots(1, 1, figsize=(3, 3))
                ax_ov.imshow(overlay)
                ax_ov.axis("off")
                fig_ov.suptitle(
                    PAPER_TAG,
                    fontdict={"family": "Times New Roman", "size": 11}
                )
                plt.tight_layout()
                overlay_path = os.path.join(overlay_dir, base_name + "_overlay.png")
                fig_ov.savefig(overlay_path, bbox_inches="tight")
                plt.close(fig_ov)

                # 清理 Grad-CAM hook 与中间变量
                cam_engine.remove_hooks()
                del cam_engine, cam_tree, heatmap, overlay
                gc.collect()

        if img_counter >= MAX_IMAGES or line_idx >= len(val_lines):
            break

    print(f"\n[完成] 共从验证集中抽取 {img_counter} 张图像做 Grad-CAM 分析。")
    print(f"结果已保存到: {os.path.abspath(SAVE_ROOT)}")


if __name__ == "__main__":
    run_gradcam_on_valset()
