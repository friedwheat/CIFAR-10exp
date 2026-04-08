"""
step2_experiments.py – 步骤二：核心实验（距离分析、分类、Bias-Variance）

包含以下三个实验：

A. 距离分析（Distance Analysis）
   - 计算各类别中心（mean）在特征空间中的 L2 距离矩阵
   - 输出并保存类间距离矩阵

B. k-NN 分类（Classification）
   - 使用 k 近邻分类器对不同特征进行分类
   - 遍历多个 k 值，记录训练集与测试集准确率

C. Bias-Variance 权衡（Bias-Variance Tradeoff）
   - 固定特征类型，遍历多个 k 值
   - 通过 bootstrap 方法近似估计 Bias² 与 Variance
   - 保存结果供可视化步骤使用

用法：
  python step2_experiments.py
  python step2_experiments.py --results_dir results --feature hist --max_train 5000
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from utils import class_means, get_class_names

# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 核心实验")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="存放特征文件与保存结果的目录（默认：results）",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="hist",
        choices=["flatten", "gray", "hist"],
        help="用于实验 B/C 的特征类型（默认：hist）",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=5000,
        help="k-NN 使用的最大训练样本数（默认：5000，减少运行时间）",
    )
    parser.add_argument(
        "--max_test",
        type=int,
        default=1000,
        help="k-NN 使用的最大测试样本数（默认：1000）",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 10, 15, 20, 30, 50],
        help="k-NN 实验使用的 k 值列表（默认：1 3 5 7 10 15 20 30 50）",
    )
    parser.add_argument(
        "--bv_bootstrap",
        type=int,
        default=10,
        help="Bias-Variance 实验的 bootstrap 轮数（默认：10）",
    )
    parser.add_argument(
        "--bv_sample_size",
        type=int,
        default=2000,
        help="每次 bootstrap 的训练样本数（默认：2000）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 工具：k-NN
# ---------------------------------------------------------------------------

def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k: int,
    batch_size: int = 200,
) -> np.ndarray:
    """基于 L2 距离的 k-NN 预测，分批计算以节省内存。

    Parameters
    ----------
    X_train  : shape (N, D)
    y_train  : shape (N,)
    X_query  : shape (M, D)
    k        : 近邻数
    batch_size : 每批查询样本数

    Returns
    -------
    y_pred : shape (M,)
    """
    M = X_query.shape[0]
    y_pred = np.empty(M, dtype=y_train.dtype)

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        # 计算 L2²：||x - y||² = ||x||² + ||y||² - 2 x·y
        q = X_query[start:end]  # (B, D)
        dists_sq = (
            np.sum(q ** 2, axis=1, keepdims=True)          # (B, 1)
            + np.sum(X_train ** 2, axis=1)                  # (N,)
            - 2.0 * q @ X_train.T                           # (B, N)
        )
        # 取前 k 个最近邻
        knn_idx = np.argpartition(dists_sq, k, axis=1)[:, :k]
        knn_labels = y_train[knn_idx]  # (B, k)
        # 多数投票
        for i in range(end - start):
            counts = np.bincount(knn_labels[i], minlength=int(y_train.max()) + 1)
            y_pred[start + i] = counts.argmax()

    return y_pred


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


# ---------------------------------------------------------------------------
# 实验 A：距离分析
# ---------------------------------------------------------------------------

def experiment_distance_analysis(
    features: Dict[str, Tuple[np.ndarray, np.ndarray]],
    results_dir: str,
) -> None:
    """计算并保存各特征类型的类间距离矩阵。"""
    print("\n" + "=" * 60)
    print("实验 A：类间距离分析")
    print("=" * 60)

    class_names = get_class_names()
    distance_matrices: Dict[str, np.ndarray] = {}

    for feat_name, (X, y) in features.items():
        means = class_means(X, y, num_classes=10)  # (10, D)
        # L2 距离矩阵
        diff = means[:, np.newaxis, :] - means[np.newaxis, :, :]  # (10, 10, D)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))            # (10, 10)
        distance_matrices[feat_name] = dist_matrix

        print(f"\n  特征：{feat_name}  |  特征维度：{X.shape[1]}")
        print(f"  {'':12}", end="")
        for name in class_names:
            print(f"{name[:6]:>8}", end="")
        print()
        for i, row_name in enumerate(class_names):
            print(f"  {row_name:<12}", end="")
            for j in range(10):
                print(f"{dist_matrix[i, j]:8.3f}", end="")
            print()

    save_path = os.path.join(results_dir, "distance_matrices.npz")
    np.savez_compressed(save_path, **distance_matrices)
    print(f"\n  距离矩阵已保存：{save_path}")


# ---------------------------------------------------------------------------
# 实验 B：k-NN 分类
# ---------------------------------------------------------------------------

def experiment_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_values: List[int],
    feat_name: str,
    results_dir: str,
) -> None:
    """遍历多个 k 值，记录训练集和测试集准确率。"""
    print("\n" + "=" * 60)
    print(f"实验 B：k-NN 分类  |  特征：{feat_name}")
    print(f"  训练集：{len(y_train)}  测试集：{len(y_test)}")
    print("=" * 60)

    train_accs, test_accs = [], []

    print(f"\n  {'k':>5}  {'训练准确率':>12}  {'测试准确率':>12}  {'耗时':>8}")
    print(f"  {'-'*45}")
    for k in k_values:
        t0 = time.time()
        y_pred_test = knn_predict(X_train, y_train, X_test, k)
        y_pred_train = knn_predict(X_train, y_train, X_train[:500], k)
        acc_train = accuracy(y_train[:500], y_pred_train)
        acc_test = accuracy(y_test, y_pred_test)
        train_accs.append(acc_train)
        test_accs.append(acc_test)
        elapsed = time.time() - t0
        print(f"  {k:>5}  {acc_train:>12.4f}  {acc_test:>12.4f}  {elapsed:>7.2f}s")

    save_path = os.path.join(results_dir, f"knn_results_{feat_name}.npz")
    np.savez_compressed(
        save_path,
        k_values=np.array(k_values),
        train_accs=np.array(train_accs),
        test_accs=np.array(test_accs),
    )
    print(f"\n  k-NN 结果已保存：{save_path}")
    best_k = k_values[int(np.argmax(test_accs))]
    print(f"  最优 k = {best_k}，测试准确率 = {max(test_accs):.4f}")


# ---------------------------------------------------------------------------
# 实验 C：Bias-Variance 权衡
# ---------------------------------------------------------------------------

def estimate_bias_variance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int,
    n_bootstrap: int,
    sample_size: int,
) -> Tuple[float, float, float]:
    """通过 bootstrap 近似估计给定 k 下的 Bias² 和 Variance。

    采用 Kohavi & Wolpert (1996) 风格的分解方式：
      - 对每个测试样本，使用多次 bootstrap 子集的预测结果
      - bias  = 平均预测与真实标签不符的比例（0/1 损失意义）
      - var   = 预测结果在不同 bootstrap 之间的方差（0/1 损失意义）
      - error = bias + var（近似）

    Returns
    -------
    bias, variance, mean_error
    """
    rng = np.random.RandomState(42)
    N_test = len(y_test)
    all_preds = np.empty((n_bootstrap, N_test), dtype=np.int64)

    for b in range(n_bootstrap):
        idx = rng.choice(len(y_train), size=sample_size, replace=True)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
        all_preds[b] = knn_predict(X_sub, y_sub, X_test, k)

    # 主预测（众数）
    n_classes = int(all_preds.max()) + 1
    main_pred = np.apply_along_axis(
        lambda col: np.bincount(col, minlength=n_classes).argmax(), axis=0, arr=all_preds
    )

    # Bias²（0/1）：主预测与真实标签不符的比例
    bias = float((main_pred != y_test).mean())

    # Variance：各次预测与主预测不符的平均比例
    variance = float((all_preds != main_pred[np.newaxis, :]).mean())

    # 平均误差
    mean_error = float((all_preds != y_test[np.newaxis, :]).mean())

    return bias, variance, mean_error


def experiment_bias_variance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_values: List[int],
    feat_name: str,
    n_bootstrap: int,
    sample_size: int,
    results_dir: str,
) -> None:
    """遍历多个 k，估计 Bias-Variance 权衡。"""
    print("\n" + "=" * 60)
    print(f"实验 C：Bias-Variance  |  特征：{feat_name}")
    print(f"  Bootstrap 轮数：{n_bootstrap}  每轮样本数：{sample_size}  测试集：{len(y_test)}")
    print("=" * 60)

    biases, variances, errors = [], [], []

    print(f"\n  {'k':>5}  {'Bias²':>10}  {'Variance':>10}  {'Error':>10}  {'耗时':>8}")
    print(f"  {'-'*50}")
    for k in k_values:
        t0 = time.time()
        b, v, e = estimate_bias_variance(
            X_train, y_train, X_test, y_test, k, n_bootstrap, sample_size
        )
        biases.append(b)
        variances.append(v)
        errors.append(e)
        elapsed = time.time() - t0
        print(f"  {k:>5}  {b:>10.4f}  {v:>10.4f}  {e:>10.4f}  {elapsed:>7.2f}s")

    save_path = os.path.join(results_dir, f"bias_variance_{feat_name}.npz")
    np.savez_compressed(
        save_path,
        k_values=np.array(k_values),
        biases=np.array(biases),
        variances=np.array(variances),
        errors=np.array(errors),
    )
    print(f"\n  Bias-Variance 结果已保存：{save_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 加载预处理特征
    # ------------------------------------------------------------------
    features_path = os.path.join(args.results_dir, "features.npz")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"找不到特征文件：{features_path}\n"
            "请先运行 step1_preprocessing.py 生成特征。"
        )

    print(f"[步骤二] 加载特征文件：{features_path}")
    data = np.load(features_path)

    all_features = {
        "flatten": (data["X_train_flat"], data["X_test_flat"]),
        "gray": (data["X_train_gray"], data["X_test_gray"]),
        "hist": (data["X_train_hist"], data["X_test_hist"]),
    }
    y_train_full = data["y_train"]
    y_test_full = data["y_test"]

    # 截取样本以加速
    N_tr = min(args.max_train, len(y_train_full))
    N_te = min(args.max_test, len(y_test_full))
    y_train = y_train_full[:N_tr]
    y_test = y_test_full[:N_te]
    print(f"  使用训练样本：{N_tr}  测试样本：{N_te}")

    # ------------------------------------------------------------------
    # 实验 A：距离分析（使用全量训练特征）
    # ------------------------------------------------------------------
    dist_features = {
        name: (data[f"X_train_{name}"], y_train_full)
        for name in ("flat", "gray", "hist")
    }
    # 键名与保存时对齐
    dist_features_renamed = {
        "flatten": (data["X_train_flat"], y_train_full),
        "gray": (data["X_train_gray"], y_train_full),
        "hist": (data["X_train_hist"], y_train_full),
    }
    experiment_distance_analysis(dist_features_renamed, args.results_dir)

    # ------------------------------------------------------------------
    # 实验 B：k-NN 分类
    # ------------------------------------------------------------------
    X_train = all_features[args.feature][0][:N_tr]
    X_test = all_features[args.feature][1][:N_te]

    experiment_classification(
        X_train, y_train,
        X_test, y_test,
        args.k_values,
        args.feature,
        args.results_dir,
    )

    # ------------------------------------------------------------------
    # 实验 C：Bias-Variance
    # ------------------------------------------------------------------
    bv_k_values = [k for k in args.k_values if k <= 20]
    experiment_bias_variance(
        X_train, y_train,
        X_test, y_test,
        bv_k_values,
        args.feature,
        n_bootstrap=args.bv_bootstrap,
        sample_size=min(args.bv_sample_size, N_tr),
        results_dir=args.results_dir,
    )

    print("\n[步骤二] 所有实验完成！\n")


if __name__ == "__main__":
    main()
