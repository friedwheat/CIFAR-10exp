"""
step3_visualization.py – 步骤三：绘图与结果展示

包含以下可视化内容：

1. 样本图像展示      – 每类随机抽取若干张图像显示
2. 类间距离矩阵热图  – 展示三种特征下的类间距离
3. k-NN 准确率曲线   – 训练集与测试集准确率随 k 变化的曲线
4. Bias-Variance 曲线 – Bias²、Variance 与总误差随 k 变化的曲线

用法：
  python step3_visualization.py
  python step3_visualization.py --results_dir results --feature hist --show
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # 无头环境下使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

from utils import get_class_names, reshape_to_hwc, to_uint8


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 结果可视化")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="存放实验结果与保存图表的目录（默认：results）",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="hist",
        choices=["flatten", "gray", "hist"],
        help="用于 k-NN / Bias-Variance 可视化的特征类型（默认：hist）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="若指定，则在保存后尝试弹出交互窗口显示图表",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1. 样本图像展示
# ---------------------------------------------------------------------------

def plot_sample_images(
    X_raw: np.ndarray,
    y: np.ndarray,
    results_dir: str,
    n_per_class: int = 5,
    show: bool = False,
) -> None:
    """为每个类别展示 n_per_class 张样本图像。"""
    class_names = get_class_names()
    n_classes = len(class_names)

    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 1.6, n_classes * 1.6),
    )

    rng = np.random.RandomState(0)
    for cls_idx, cls_name in enumerate(class_names):
        cls_indices = np.where(y == cls_idx)[0]
        chosen = rng.choice(cls_indices, size=min(n_per_class, len(cls_indices)), replace=False)
        for col, img_idx in enumerate(chosen):
            img = reshape_to_hwc(X_raw[img_idx : img_idx + 1])[0]  # (32, 32, 3)
            img = to_uint8(img / 255.0)
            axes[cls_idx, col].imshow(img)
            axes[cls_idx, col].axis("off")
            if col == 0:
                axes[cls_idx, col].set_ylabel(cls_name, fontsize=9, rotation=0,
                                               labelpad=50, va="center")

    fig.suptitle("CIFAR-10 各类别样本图像", fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "sample_images.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    print(f"  [图1] 样本图像已保存：{save_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 2. 类间距离矩阵热图
# ---------------------------------------------------------------------------

def plot_distance_matrices(
    results_dir: str,
    show: bool = False,
) -> None:
    """绘制三种特征的类间距离矩阵热图。"""
    dist_path = os.path.join(results_dir, "distance_matrices.npz")
    if not os.path.exists(dist_path):
        print(f"  [图2] 未找到距离矩阵文件：{dist_path}，跳过。")
        return

    data = np.load(dist_path)
    class_names = get_class_names()
    feat_names = list(data.keys())
    n_feats = len(feat_names)

    fig, axes = plt.subplots(1, n_feats, figsize=(6 * n_feats, 5))
    if n_feats == 1:
        axes = [axes]

    for ax, feat in zip(axes, feat_names):
        mat = data[feat]
        n = mat.shape[0]  # 类别数，由矩阵维度推导
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names[:n], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names[:n], fontsize=8)
        ax.set_title(f"特征：{feat}", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)
        # 在格子内标注数值
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:.1f}",
                        ha="center", va="center", fontsize=5,
                        color="white" if mat[i, j] > mat.max() * 0.5 else "black")

    fig.suptitle("CIFAR-10 各类别间 L2 距离矩阵", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "distance_matrices.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    print(f"  [图2] 距离矩阵热图已保存：{save_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 3. k-NN 准确率曲线
# ---------------------------------------------------------------------------

def plot_knn_accuracy(
    feature: str,
    results_dir: str,
    show: bool = False,
) -> None:
    """绘制 k-NN 训练集与测试集准确率随 k 变化的曲线。"""
    knn_path = os.path.join(results_dir, f"knn_results_{feature}.npz")
    if not os.path.exists(knn_path):
        print(f"  [图3] 未找到 k-NN 结果文件：{knn_path}，跳过。")
        return

    data = np.load(knn_path)
    k_values = data["k_values"]
    train_accs = data["train_accs"]
    test_accs = data["test_accs"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, train_accs, "o-", label="训练集准确率", color="steelblue")
    ax.plot(k_values, test_accs, "s--", label="测试集准确率", color="tomato")

    best_k_idx = int(np.argmax(test_accs))
    ax.axvline(k_values[best_k_idx], color="gray", linestyle=":", alpha=0.7,
               label=f"最优 k={k_values[best_k_idx]}（测试={test_accs[best_k_idx]:.3f}）")

    ax.set_xlabel("k（近邻数）", fontsize=12)
    ax.set_ylabel("准确率", fontsize=12)
    ax.set_title(f"k-NN 准确率随 k 变化（特征：{feature}）", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = os.path.join(results_dir, f"knn_accuracy_{feature}.png")
    plt.savefig(save_path, dpi=120)
    print(f"  [图3] k-NN 准确率曲线已保存：{save_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 4. Bias-Variance 曲线
# ---------------------------------------------------------------------------

def plot_bias_variance(
    feature: str,
    results_dir: str,
    show: bool = False,
) -> None:
    """绘制 Bias²、Variance 与总误差随 k 变化的曲线。"""
    bv_path = os.path.join(results_dir, f"bias_variance_{feature}.npz")
    if not os.path.exists(bv_path):
        print(f"  [图4] 未找到 Bias-Variance 结果文件：{bv_path}，跳过。")
        return

    data = np.load(bv_path)
    k_values = data["k_values"]
    biases = data["biases"]
    variances = data["variances"]
    errors = data["errors"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, biases, "o-", label="Bias²（偏差）", color="tomato")
    ax.plot(k_values, variances, "s-", label="Variance（方差）", color="steelblue")
    ax.plot(k_values, errors, "^--", label="总误差", color="darkorange")

    ax.set_xlabel("k（近邻数）", fontsize=12)
    ax.set_ylabel("误差", fontsize=12)
    ax.set_title(f"Bias-Variance 权衡（特征：{feature}）", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = os.path.join(results_dir, f"bias_variance_{feature}.png")
    plt.savefig(save_path, dpi=120)
    print(f"  [图4] Bias-Variance 曲线已保存：{save_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 5. 综合摘要图（可选）
# ---------------------------------------------------------------------------

def plot_summary(
    feature: str,
    results_dir: str,
    show: bool = False,
) -> None:
    """将 k-NN 准确率与 Bias-Variance 合并为一张摘要图。"""
    knn_path = os.path.join(results_dir, f"knn_results_{feature}.npz")
    bv_path = os.path.join(results_dir, f"bias_variance_{feature}.npz")

    if not os.path.exists(knn_path) or not os.path.exists(bv_path):
        print("  [图5] 缺少数据文件，跳过摘要图。")
        return

    knn_data = np.load(knn_path)
    bv_data = np.load(bv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：准确率
    ax1.plot(knn_data["k_values"], knn_data["train_accs"], "o-",
             label="训练集", color="steelblue")
    ax1.plot(knn_data["k_values"], knn_data["test_accs"], "s--",
             label="测试集", color="tomato")
    ax1.set_xlabel("k", fontsize=11)
    ax1.set_ylabel("准确率", fontsize=11)
    ax1.set_title(f"k-NN 准确率（{feature}）")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # 右图：Bias-Variance
    ax2.plot(bv_data["k_values"], bv_data["biases"], "o-",
             label="Bias²", color="tomato")
    ax2.plot(bv_data["k_values"], bv_data["variances"], "s-",
             label="Variance", color="steelblue")
    ax2.plot(bv_data["k_values"], bv_data["errors"], "^--",
             label="总误差", color="darkorange")
    ax2.set_xlabel("k", fontsize=11)
    ax2.set_ylabel("误差", fontsize=11)
    ax2.set_title(f"Bias-Variance 权衡（{feature}）")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.suptitle("CIFAR-10 k-NN 实验综合摘要", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"summary_{feature}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    print(f"  [图5] 综合摘要图已保存：{save_path}")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"[步骤三] 结果目录：{args.results_dir}")
    print(f"[步骤三] 特征类型：{args.feature}\n")

    # 图1：样本图像
    raw_path = os.path.join(args.results_dir, "raw_images.npz")
    if os.path.exists(raw_path):
        raw = np.load(raw_path)
        plot_sample_images(
            raw["X_train_raw"], raw["y_train"],
            args.results_dir, n_per_class=5, show=args.show,
        )
    else:
        print(f"  [图1] 未找到原始图像文件：{raw_path}，跳过。")

    # 图2：距离矩阵热图
    plot_distance_matrices(args.results_dir, show=args.show)

    # 图3：k-NN 准确率曲线
    plot_knn_accuracy(args.feature, args.results_dir, show=args.show)

    # 图4：Bias-Variance 曲线
    plot_bias_variance(args.feature, args.results_dir, show=args.show)

    # 图5：综合摘要图
    plot_summary(args.feature, args.results_dir, show=args.show)

    print("\n[步骤三] 所有图表生成完毕！\n")


if __name__ == "__main__":
    main()
