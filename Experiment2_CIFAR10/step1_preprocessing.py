"""
step1_preprocessing.py – 步骤一：数据读取、采样与多尺度特征构造

实现内容：
  1. 从 data/ 目录读取 CIFAR-10 原始批次文件
  2. 将 1×3072 向量转换为 32×32×3 图像，并执行 RGB->BGR 转换
  3. 分层采样：训练集每类 200，测试集每类 50
  4. 构造 7 种维度特征：[16, 32, 64, 128, 256, 512, 3072]
     - 使用 cv2.cvtColor 转灰度
     - 使用 cv2.resize 缩放
     - Flatten 成特征向量
"""

import argparse
import os
import pickle
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np


CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

FEATURE_DIMS: List[int] = [16, 32, 64, 128, 256, 512, 3072]
# cv2.resize 使用 (width, height)
GRAY_RESIZE_SHAPES: Dict[int, Tuple[int, int]] = {
    16: (4, 4),
    32: (8, 4),
    64: (8, 8),
    128: (16, 8),
    256: (16, 16),
    512: (32, 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 数据预处理与特征提取")
    parser.add_argument("--data_dir", type=str, default="data", help="CIFAR-10 批次文件目录")
    parser.add_argument("--results_dir", type=str, default="results", help="结果保存目录")
    parser.add_argument("--train_per_class", type=int, default=200, help="训练集每类采样数量")
    parser.add_argument("--test_per_class", type=int, default=50, help="测试集每类采样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def load_cifar10_batch(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(filepath, "rb") as f:
        entry = pickle.load(f, encoding="bytes")
    X = entry[b"data"].astype(np.uint8)  # (N, 3072)
    y = np.array(entry[b"labels"], dtype=np.int64)
    return X, y


def load_cifar10_from_data_dir(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_list, y_train_list = [], []
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X_batch, y_batch = load_cifar10_batch(batch_path)
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    test_path = os.path.join(data_dir, "test_batch")
    X_test, y_test = load_cifar10_batch(test_path)
    return X_train, y_train, X_test, y_test


def vec3072_to_rgb(vec: np.ndarray) -> np.ndarray:
    return vec.reshape(3, 32, 32).transpose(1, 2, 0)


def vec3072_to_bgr(vec: np.ndarray) -> np.ndarray:
    rgb = vec3072_to_rgb(vec)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def stratified_sample_per_class(
    X: np.ndarray,
    y: np.ndarray,
    n_per_class: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    sampled_idx = []
    for cls in range(10):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) < n_per_class:
            raise ValueError(f"类别 {cls} 样本不足：需要 {n_per_class}，实际 {len(cls_idx)}")
        chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
        sampled_idx.append(chosen)

    sampled_idx = np.concatenate(sampled_idx)
    rng.shuffle(sampled_idx)
    return X[sampled_idx], y[sampled_idx]


def extract_feature_from_bgr(img_bgr: np.ndarray, dim: int) -> np.ndarray:
    if dim == 3072:
        return img_bgr.reshape(-1).astype(np.float32) / 255.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    width, height = GRAY_RESIZE_SHAPES[dim]
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    return resized.reshape(-1).astype(np.float32) / 255.0


def extract_multiscale_features(X_vec: np.ndarray) -> Dict[int, np.ndarray]:
    features: Dict[int, np.ndarray] = {}
    for dim in FEATURE_DIMS:
        features[dim] = np.empty((len(X_vec), dim), dtype=np.float32)

    for i in range(len(X_vec)):
        img_bgr = vec3072_to_bgr(X_vec[i])
        for dim in FEATURE_DIMS:
            features[dim][i] = extract_feature_from_bgr(img_bgr, dim)

    return features


def print_class_stats(y_train: np.ndarray, y_test: np.ndarray) -> None:
    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)
    print(f"  {'类别':<12} {'训练集数量':>10} {'测试集数量':>10}")
    print(f"  {'-'*34}")
    for i, name in enumerate(CIFAR10_CLASSES):
        print(f"  {name:<12} {train_counts[i]:>10} {test_counts[i]:>10}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"[步骤一] 数据目录：{args.data_dir}")
    print(f"[步骤一] 结果目录：{args.results_dir}")

    print("\n[1/4] 加载 CIFAR-10 原始数据 ...")
    t0 = time.time()
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_cifar10_from_data_dir(args.data_dir)
    print(
        f"  训练集：{X_train_raw.shape[0]} 张图像  |  "
        f"测试集：{X_test_raw.shape[0]} 张图像  |  "
        f"耗时：{time.time() - t0:.2f}s"
    )

    print("\n[2/4] 分层采样 ...")
    rng = np.random.RandomState(args.seed)
    X_train, y_train = stratified_sample_per_class(
        X_train_raw, y_train_raw, n_per_class=args.train_per_class, rng=rng
    )
    X_test, y_test = stratified_sample_per_class(
        X_test_raw, y_test_raw, n_per_class=args.test_per_class, rng=rng
    )
    print(f"  采样后训练集：{len(y_train)}（每类 {args.train_per_class}）")
    print(f"  采样后测试集：{len(y_test)}（每类 {args.test_per_class}）")

    print("\n[3/4] 构造 7 种维度特征 ...")
    t0 = time.time()
    train_features = extract_multiscale_features(X_train)
    test_features = extract_multiscale_features(X_test)
    for dim in FEATURE_DIMS:
        print(f"  [dim={dim:>4}] train={train_features[dim].shape} test={test_features[dim].shape}")
    print(f"  特征提取耗时：{time.time() - t0:.2f}s")

    print("\n[4/4] 保存结果 ...")
    save_dict = {
        "y_train": y_train,
        "y_test": y_test,
        # 兼容键（保持 step2 默认 flatten 可直接使用）
        "X_train_flat": train_features[3072],
        "X_test_flat": test_features[3072],
    }
    for dim in FEATURE_DIMS:
        save_dict[f"X_train_{dim}"] = train_features[dim]
        save_dict[f"X_test_{dim}"] = test_features[dim]

    features_path = os.path.join(args.results_dir, "features.npz")
    np.savez_compressed(features_path, **save_dict)
    print(f"  已保存：{features_path}")

    raw_path = os.path.join(args.results_dir, "raw_images.npz")
    np.savez_compressed(
        raw_path,
        X_train_raw=X_train,
        y_train=y_train,
        X_test_raw=X_test,
        y_test=y_test,
    )
    print(f"  已保存：{raw_path}")

    print("\n采样后数据统计：")
    print_class_stats(y_train, y_test)
    print("\n[步骤一] 完成！\n")


if __name__ == "__main__":
    main()
