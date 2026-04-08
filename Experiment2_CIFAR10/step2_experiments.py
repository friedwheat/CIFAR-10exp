"""
step2_experiments.py – 步骤二：多分辨率下的 1-NN / k-NN 实验

实验内容：
1) 1-NN 最近邻距离分析（Nearest / Average Distance）
2) 1-NN 分类与错例诊断
3) k=1,3,5 的分类误差曲线对比
4) 固定两类二分类任务：重复采样训练子集，估计 MSE / Variance / Sq. Bias
"""

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIMS = [16, 64, 144, 256, 576, 1024, 3072]
FULL_RES_DIM = 3072
EPSILON = 1e-12
PIXEL_MAX_VALUE = 255.0
IMAGE_SIDE = 32
NUM_CHANNELS = 3
DIAG_FIGSIZE = (6.5, 3.2)
BINARY_DEFAULT_CLASSES = [0, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 多分辨率 1-NN / k-NN 实验")
    parser.add_argument("--results_dir", type=str, default="results", help="结果目录（默认：results）")
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=DEFAULT_DIMS,
        help=f"参与实验的特征维度（默认：{' '.join(map(str, DEFAULT_DIMS))}）",
    )
    parser.add_argument("--max_train", type=int, default=2000, help="训练样本上限（默认：2000）")
    parser.add_argument("--max_test", type=int, default=500, help="测试样本上限（默认：500）")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5], help="k 值列表（默认：1 3 5）")
    parser.add_argument("--batch_size", type=int, default=100, help="分批计算大小（默认：100）")
    parser.add_argument(
        "--binary_classes",
        type=int,
        nargs=2,
        default=BINARY_DEFAULT_CLASSES,
        metavar=("CLASS_A", "CLASS_B"),
        help="二分类固定类别（默认：0 1）",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=50,
        help="重复实验轮数（生成独立训练子集数量，默认：50）",
    )
    parser.add_argument(
        "--subset_train_size",
        type=int,
        default=400,
        help="每轮随机采样训练样本数（不放回，默认：400）",
    )
    parser.add_argument("--experiment_seed", type=int, default=42, help="重复实验随机种子（默认：42）")
    return parser.parse_args()


def l2_distances(query: np.ndarray, train: np.ndarray) -> np.ndarray:
    """返回 query 到 train 的 L2 距离矩阵，shape=(B, N)。"""
    d2 = (
        np.sum(query ** 2, axis=1, keepdims=True)
        + np.sum(train ** 2, axis=1)
        - 2.0 * query @ train.T
    )
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


def evaluate_distances_and_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k_values: Sequence[int],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """计算测试集上的最近邻距离、平均距离、最近邻索引与各 k 预测。"""
    n_test = X_test.shape[0]
    k_values = sorted(set(int(k) for k in k_values))
    max_k = max(k_values)

    nearest_dist = np.empty(n_test, dtype=np.float64)
    avg_dist = np.empty(n_test, dtype=np.float64)
    nearest_idx = np.empty(n_test, dtype=np.int64)
    preds: Dict[int, np.ndarray] = {k: np.empty(n_test, dtype=y_train.dtype) for k in k_values}
    n_classes = int(y_train.max()) + 1

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        dist = l2_distances(X_test[start:end], X_train)

        local_nn_idx = np.argmin(dist, axis=1)
        nearest_idx[start:end] = local_nn_idx
        nearest_dist[start:end] = dist[np.arange(end - start), local_nn_idx]
        avg_dist[start:end] = dist.mean(axis=1)

        nn_part_idx = np.argpartition(dist, kth=max_k - 1, axis=1)[:, :max_k]
        nn_part_dist = np.take_along_axis(dist, nn_part_idx, axis=1)
        order = np.argsort(nn_part_dist, axis=1)
        nn_idx_sorted = np.take_along_axis(nn_part_idx, order, axis=1)
        nn_labels = y_train[nn_idx_sorted]

        for k in k_values:
            topk = nn_labels[:, :k]
            for i in range(end - start):
                counts = np.bincount(topk[i], minlength=n_classes)
                preds[k][start + i] = counts.argmax()

    return nearest_dist, avg_dist, nearest_idx, preds


def plot_distance_curves(
    dims: List[int],
    nearest_means: List[float],
    ratio_means: List[float],
    results_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(dims, nearest_means, "o-", color="royalblue")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Average Distance to 1-NN")
    axes[0].set_title("Average Distance to 1-NN vs Dimension")
    axes[0].grid(alpha=0.3)

    axes[1].plot(dims, ratio_means, "s-", color="tomato")
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Distance Ratio (Nearest/Average)")
    axes[1].set_title("Distance Ratio (Nearest/Average) vs Dimension")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "distance_analysis_vs_dimension.png")
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  距离分析曲线已保存：{save_path}")


def plot_knn_error_curves(
    dims: List[int],
    error_by_k: Dict[int, List[float]],
    results_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {1: "o-", 3: "s--", 5: "^-."}
    for k, errors in sorted(error_by_k.items()):
        ax.plot(dims, errors, markers.get(k, "o-"), label=f"k={k}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Classification Error")
    ax.set_title("k-NN Classification Error vs Dimension")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "knn_error_vs_dimension.png")
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  分类误差曲线已保存：{save_path}")


def generate_subset_indices(
    pool_size: int,
    num_repeats: int,
    subset_train_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    if pool_size <= 0:
        raise ValueError("训练池为空，无法生成训练子集。")
    if num_repeats <= 0:
        raise ValueError("num_repeats 必须为正整数。")
    if subset_train_size <= 0:
        raise ValueError("subset_train_size 必须为正整数。")
    if subset_train_size > pool_size:
        raise ValueError(
            f"subset_train_size={subset_train_size} 大于训练池大小 {pool_size}，无法在单轮内无放回采样。"
        )

    subset_indices = np.empty((num_repeats, subset_train_size), dtype=np.int64)
    all_idx = np.arange(pool_size)
    for r in range(num_repeats):
        subset_indices[r] = rng.choice(all_idx, size=subset_train_size, replace=False)
    return subset_indices


def predict_1nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    preds = np.empty(X_test.shape[0], dtype=np.int64)
    for start in range(0, X_test.shape[0], batch_size):
        end = min(start + batch_size, X_test.shape[0])
        dist = l2_distances(X_test[start:end], X_train)
        nn_idx = np.argmin(dist, axis=1)
        preds[start:end] = y_train[nn_idx]
    return preds


def ensure_float32(X: np.ndarray) -> np.ndarray:
    return X if X.dtype == np.float32 else X.astype(np.float32)


def evaluate_binary_bias_variance(
    X_train: np.ndarray,
    y_train_bin: np.ndarray,
    X_test: np.ndarray,
    y_test_bin: np.ndarray,
    subset_indices: np.ndarray,
    batch_size: int,
) -> Tuple[float, float, float]:
    num_repeats = subset_indices.shape[0]
    preds = np.empty((num_repeats, X_test.shape[0]), dtype=np.int64)
    y_test_float = y_test_bin.astype(np.float32)

    for r in range(num_repeats):
        idx = subset_indices[r]
        preds[r] = predict_1nn(
            X_train=X_train[idx],
            y_train=y_train_bin[idx],
            X_test=X_test,
            batch_size=batch_size,
        )

    preds_float = preds.astype(np.float32)
    mean_pred = preds_float.mean(axis=0)
    # Variance = E_x[Var_D(f_D(x))]
    variance = float(preds_float.var(axis=0).mean())
    # Sq. Bias = E_x[(E_D[f_D(x)] - y(x))^2]
    sq_bias = float(((mean_pred - y_test_float) ** 2).mean())
    mse = float(((preds_float - y_test_float[None, :]) ** 2).mean())
    return mse, variance, sq_bias


def plot_mse_variance_bias_curves(
    dims: List[int],
    mses: List[float],
    variances: List[float],
    sq_biases: List[float],
    results_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, mses, "o-", label="MSE", color="darkorange")
    ax.plot(dims, variances, "s-", label="Variance", color="steelblue")
    ax.plot(dims, sq_biases, "^-", label="Sq. Bias", color="tomato")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Error")
    ax.set_title("MSE / Variance / Sq. Bias vs Dimension (1-NN Binary)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    save_path = os.path.join(results_dir, "mse_variance_sq_bias_vs_dimension.png")
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  MSE/Variance/Sq. Bias 曲线已保存：{save_path}")


def vec3072_to_rgb(vec: np.ndarray) -> np.ndarray:
    return vec.reshape(NUM_CHANNELS, IMAGE_SIDE, IMAGE_SIDE).transpose(1, 2, 0).astype(np.uint8)


def load_images_for_diagnosis(
    results_dir: str,
    features_data: np.lib.npyio.NpzFile,
    n_train: int,
    n_test: int,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_path = os.path.join(results_dir, "raw_images.npz")
    if os.path.exists(raw_path):
        raw = np.load(raw_path)
        return raw["X_train_raw"][:n_train], raw["X_test_raw"][:n_test]

    # 兜底使用 step1 生成的 3072 维特征（按 BGR 顺序 flatten，显示前转回 RGB）
    if "X_train_3072" not in features_data or "X_test_3072" not in features_data:
        raise KeyError("raw_images.npz 缺失且 features.npz 中无 X_train_3072/X_test_3072，无法生成错例图。")
    train_bgr = (
        (features_data["X_train_3072"][:n_train] * PIXEL_MAX_VALUE)
        .astype(np.uint8)
        .reshape(-1, IMAGE_SIDE, IMAGE_SIDE, NUM_CHANNELS)
    )
    test_bgr = (
        (features_data["X_test_3072"][:n_test] * PIXEL_MAX_VALUE)
        .astype(np.uint8)
        .reshape(-1, IMAGE_SIDE, IMAGE_SIDE, NUM_CHANNELS)
    )
    train_rgb = train_bgr[:, :, :, ::-1].reshape(-1, FULL_RES_DIM)
    test_rgb = test_bgr[:, :, :, ::-1].reshape(-1, FULL_RES_DIM)
    return train_rgb, test_rgb


def plot_misclassified_pair(
    dim: int,
    test_idx: int,
    nn_train_idx: int,
    y_true: int,
    y_pred: int,
    train_raw: np.ndarray,
    test_raw: np.ndarray,
    results_dir: str,
) -> None:
    test_img = vec3072_to_rgb(test_raw[test_idx])
    nn_img = vec3072_to_rgb(train_raw[nn_train_idx])

    fig, axes = plt.subplots(1, 2, figsize=DIAG_FIGSIZE)
    axes[0].imshow(test_img)
    axes[0].set_title(f"Test (idx={test_idx})\ntrue={y_true}, pred={y_pred}")
    axes[0].axis("off")

    axes[1].imshow(nn_img)
    axes[1].set_title(f"Nearest Train (idx={nn_train_idx})\nlabel={y_pred}")
    axes[1].axis("off")

    fig.suptitle(f"Misclassified 1-NN Pair @ dim={dim}")
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"misclassified_pair_dim_{dim}.png")
    plt.savefig(save_path, dpi=140)
    plt.close()
    print(f"  错例图已保存：{save_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    features_path = os.path.join(args.results_dir, "features.npz")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"找不到特征文件：{features_path}\n请先运行 step1_preprocessing.py 生成特征。"
        )

    print(f"[步骤二] 加载特征：{features_path}")
    data = np.load(features_path)
    y_train_full = data["y_train"]
    y_test_full = data["y_test"]

    dims = []
    for dim in sorted(set(args.dims)):
        if f"X_train_{dim}" in data and f"X_test_{dim}" in data:
            dims.append(dim)
        else:
            print(f"  [警告] 未找到 dim={dim} 对应特征，已跳过。")
    if not dims:
        raise ValueError("没有可用的维度特征，请检查 --dims 参数与 features.npz 内容。")

    n_train = min(args.max_train, len(y_train_full))
    n_test = min(args.max_test, len(y_test_full))
    y_train = y_train_full[:n_train]
    y_test = y_test_full[:n_test]
    k_values = sorted(set(args.k_values))
    if 1 not in k_values:
        print("  [警告] k 列表中未包含 1，已自动加入 k=1 以支持 1-NN 距离分析与错例诊断。")
        k_values = sorted(set([1] + k_values))

    print(f"  使用样本：train={n_train} test={n_test}")
    print(f"  维度列表：{dims}")
    print(f"  k 列表：{k_values}")

    nearest_mean_by_dim: List[float] = []
    avg_mean_by_dim: List[float] = []
    ratio_mean_by_dim: List[float] = []
    error_by_k: Dict[int, List[float]] = {k: [] for k in k_values}
    pred_by_dim_k1: Dict[int, np.ndarray] = {}
    nn_idx_by_dim: Dict[int, np.ndarray] = {}

    print("\n" + "=" * 78)
    print("实验 1+2+3：距离分析 + 1-NN/k-NN 分类")
    print("=" * 78)
    print(
        f"{'dim':>6}  {'near_mean':>12}  {'avg_mean':>12}  {'ratio_mean':>12}  "
        + "  ".join([f"err@k={k:>2}" for k in sorted(error_by_k)])
    )
    print("-" * 78)

    for dim in dims:
        X_train = data[f"X_train_{dim}"][:n_train]
        X_test = data[f"X_test_{dim}"][:n_test]
        if X_train.dtype != np.float32:
            X_train = X_train.astype(np.float32)
        if X_test.dtype != np.float32:
            X_test = X_test.astype(np.float32)

        nearest_dist, avg_dist, nearest_idx, preds = evaluate_distances_and_knn(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            k_values=sorted(error_by_k.keys()),
            batch_size=args.batch_size,
        )

        ratio = nearest_dist / np.maximum(avg_dist, EPSILON)
        nearest_mean = float(nearest_dist.mean())
        avg_mean = float(avg_dist.mean())
        ratio_mean = float(ratio.mean())

        nearest_mean_by_dim.append(nearest_mean)
        avg_mean_by_dim.append(avg_mean)
        ratio_mean_by_dim.append(ratio_mean)
        pred_by_dim_k1[dim] = preds[1]
        nn_idx_by_dim[dim] = nearest_idx

        err_parts = []
        for k in sorted(error_by_k):
            err = float((preds[k] != y_test).mean())
            error_by_k[k].append(err)
            err_parts.append(f"{err:>8.4f}")

        print(f"{dim:>6}  {nearest_mean:>12.6f}  {avg_mean:>12.6f}  {ratio_mean:>12.6f}  " + "  ".join(err_parts))

    print("-" * 78)

    plot_distance_curves(dims, nearest_mean_by_dim, ratio_mean_by_dim, args.results_dir)
    plot_knn_error_curves(dims, error_by_k, args.results_dir)

    print("\n" + "=" * 78)
    print("附加实验：固定两类二分类 + 重复采样训练子集的 1-NN Bias-Variance 分解")
    print("=" * 78)
    class_a, class_b = int(args.binary_classes[0]), int(args.binary_classes[1])
    if class_a == class_b:
        raise ValueError("--binary_classes 需要提供两个不同类别。")

    train_bin_mask = np.logical_or(y_train_full == class_a, y_train_full == class_b)
    test_bin_mask = np.logical_or(y_test_full == class_a, y_test_full == class_b)
    if not train_bin_mask.any() or not test_bin_mask.any():
        raise ValueError(
            f"指定类别 {class_a}/{class_b} 在数据中不存在有效样本，请检查 --binary_classes。"
        )

    y_train_bin = (y_train_full[train_bin_mask] == class_b).astype(np.int64)
    y_test_bin = (y_test_full[test_bin_mask] == class_b).astype(np.int64)
    rng = np.random.RandomState(args.experiment_seed)
    subset_indices = generate_subset_indices(
        pool_size=len(y_train_bin),
        num_repeats=args.num_repeats,
        subset_train_size=args.subset_train_size,
        rng=rng,
    )

    print(
        f"  二分类类别: {class_a}->0, {class_b}->1 | "
        f"train_pool={len(y_train_bin)} test_pool={len(y_test_bin)} | "
        f"repeats={args.num_repeats} subset_size={args.subset_train_size}"
    )
    print(f"{'dim':>6}  {'MSE':>10}  {'Variance':>10}  {'Sq. Bias':>10}  {'MSE-(B+V)':>12}")
    print("-" * 78)

    mse_by_dim: List[float] = []
    variance_by_dim: List[float] = []
    sq_bias_by_dim: List[float] = []
    for dim in dims:
        X_train_bin = ensure_float32(data[f"X_train_{dim}"][train_bin_mask])
        X_test_bin = ensure_float32(data[f"X_test_{dim}"][test_bin_mask])

        mse, variance, sq_bias = evaluate_binary_bias_variance(
            X_train=X_train_bin,
            y_train_bin=y_train_bin,
            X_test=X_test_bin,
            y_test_bin=y_test_bin,
            subset_indices=subset_indices,
            batch_size=args.batch_size,
        )
        mse_by_dim.append(mse)
        variance_by_dim.append(variance)
        sq_bias_by_dim.append(sq_bias)
        residual = mse - (variance + sq_bias)
        print(f"{dim:>6}  {mse:>10.6f}  {variance:>10.6f}  {sq_bias:>10.6f}  {residual:>12.6f}")
    print("-" * 78)

    plot_mse_variance_bias_curves(dims, mse_by_dim, variance_by_dim, sq_bias_by_dim, args.results_dir)

    # 错例：优先最高维（3072）；若该维度无错例，则退化到 k=1 错误率最高的维度
    dim_for_case = FULL_RES_DIM if FULL_RES_DIM in dims else max(dims)
    mis_idx = np.where(pred_by_dim_k1[dim_for_case] != y_test)[0]
    if mis_idx.size == 0:
        if 1 in error_by_k:
            worst_dim_idx = int(np.argmax(error_by_k[1]))
            dim_for_case = dims[worst_dim_idx]
            mis_idx = np.where(pred_by_dim_k1[dim_for_case] != y_test)[0]

    if mis_idx.size > 0:
        chosen_test_idx = int(mis_idx[0])
        chosen_nn_idx = int(nn_idx_by_dim[dim_for_case][chosen_test_idx])
        train_raw, test_raw = load_images_for_diagnosis(args.results_dir, data, n_train, n_test)
        plot_misclassified_pair(
            dim=dim_for_case,
            test_idx=chosen_test_idx,
            nn_train_idx=chosen_nn_idx,
            y_true=int(y_test[chosen_test_idx]),
            y_pred=int(pred_by_dim_k1[dim_for_case][chosen_test_idx]),
            train_raw=train_raw,
            test_raw=test_raw,
            results_dir=args.results_dir,
        )
    else:
        print("  [提示] 所有维度在 k=1 上均无错例，跳过错例图。")

    save_path = os.path.join(args.results_dir, "nn_knn_multidim_results.npz")
    save_dict = {
        "dims": np.array(dims),
        "nearest_mean": np.array(nearest_mean_by_dim),
        "average_mean": np.array(avg_mean_by_dim),
        "ratio_mean": np.array(ratio_mean_by_dim),
    }
    for k, errs in error_by_k.items():
        save_dict[f"error_k_{k}"] = np.array(errs)
    save_dict["binary_classes"] = np.array([class_a, class_b], dtype=np.int64)
    save_dict["binary_num_repeats"] = args.num_repeats
    save_dict["binary_subset_train_size"] = args.subset_train_size
    save_dict["binary_mse"] = np.array(mse_by_dim, dtype=np.float64)
    save_dict["binary_variance"] = np.array(variance_by_dim, dtype=np.float64)
    save_dict["binary_sq_bias"] = np.array(sq_bias_by_dim, dtype=np.float64)
    np.savez_compressed(save_path, **save_dict)
    print(f"  结果数据已保存：{save_path}")
    print("\n[步骤二] 实验完成！\n")


if __name__ == "__main__":
    main()
