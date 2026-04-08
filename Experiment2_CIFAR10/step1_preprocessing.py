"""
step1_preprocessing.py – 步骤一：数据读取与特征提取

功能：
  1. 从 data/ 目录读取 CIFAR-10 原始批次文件。
  2. 对图像进行归一化预处理。
  3. 提取三种特征：
       - flatten : 展平的归一化 RGB 像素（3072 维）
       - gray    : 灰度展平像素（1024 维）
       - hist    : RGB 颜色直方图（96 维，每通道 32 bin）
  4. 将处理后的特征与标签保存到 results/ 目录，供后续步骤使用。

用法：
  python step1_preprocessing.py
  python step1_preprocessing.py --data_dir data --results_dir results
"""

import argparse
import os
import time

import numpy as np

from utils import (
    extract_color_histogram,
    extract_flatten,
    extract_grayscale_flatten,
    get_class_names,
    load_cifar10,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 数据预处理与特征提取")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="存放 CIFAR-10 原始批次文件的目录（默认：data）",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="保存处理结果的目录（默认：results）",
    )
    parser.add_argument(
        "--num_train_batches",
        type=int,
        default=5,
        help="要加载的训练批次数，取值 1–5（默认：5）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 0. 准备目录
    # ------------------------------------------------------------------
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"[步骤一] 数据目录：{args.data_dir}")
    print(f"[步骤一] 结果目录：{args.results_dir}")

    # ------------------------------------------------------------------
    # 1. 加载原始数据
    # ------------------------------------------------------------------
    print("\n[1/4] 加载 CIFAR-10 原始数据 ...")
    t0 = time.time()
    X_train_raw, y_train, X_test_raw, y_test = load_cifar10(
        args.data_dir, num_train_batches=args.num_train_batches
    )
    print(
        f"  训练集：{X_train_raw.shape[0]} 张图像  |  "
        f"测试集：{X_test_raw.shape[0]} 张图像  |  "
        f"耗时：{time.time() - t0:.2f}s"
    )
    print(f"  类别：{get_class_names()}")

    # ------------------------------------------------------------------
    # 2. 特征提取
    # ------------------------------------------------------------------
    print("\n[2/4] 提取特征 ...")

    # (a) 展平 RGB 像素
    t0 = time.time()
    X_train_flat = extract_flatten(X_train_raw)
    X_test_flat = extract_flatten(X_test_raw)
    # 对展平特征进行零均值标准化
    X_train_flat, flat_mean, flat_std = standardize(X_train_flat)
    X_test_flat, _, _ = standardize(X_test_flat, mean=flat_mean, std=flat_std)
    print(f"  [flatten] 维度 {X_train_flat.shape[1]}  耗时 {time.time() - t0:.2f}s")

    # (b) 灰度展平
    t0 = time.time()
    X_train_gray = extract_grayscale_flatten(X_train_raw)
    X_test_gray = extract_grayscale_flatten(X_test_raw)
    X_train_gray, gray_mean, gray_std = standardize(X_train_gray)
    X_test_gray, _, _ = standardize(X_test_gray, mean=gray_mean, std=gray_std)
    print(f"  [gray   ] 维度 {X_train_gray.shape[1]}  耗时 {time.time() - t0:.2f}s")

    # (c) 颜色直方图
    t0 = time.time()
    X_train_hist = extract_color_histogram(X_train_raw, bins=32)
    X_test_hist = extract_color_histogram(X_test_raw, bins=32)
    X_train_hist, hist_mean, hist_std = standardize(X_train_hist)
    X_test_hist, _, _ = standardize(X_test_hist, mean=hist_mean, std=hist_std)
    print(f"  [hist   ] 维度 {X_train_hist.shape[1]}  耗时 {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 3. 保存结果
    # ------------------------------------------------------------------
    print("\n[3/4] 保存特征与标签到 numpy 文件 ...")
    save_path = os.path.join(args.results_dir, "features.npz")
    np.savez_compressed(
        save_path,
        X_train_flat=X_train_flat,
        X_test_flat=X_test_flat,
        X_train_gray=X_train_gray,
        X_test_gray=X_test_gray,
        X_train_hist=X_train_hist,
        X_test_hist=X_test_hist,
        y_train=y_train,
        y_test=y_test,
    )
    print(f"  已保存：{save_path}")

    # 同时保存用于可视化的原始图像（仅训练集前 500 张，节省空间）
    vis_path = os.path.join(args.results_dir, "raw_images.npz")
    np.savez_compressed(
        vis_path,
        X_train_raw=X_train_raw[:500],
        y_train=y_train[:500],
        X_test_raw=X_test_raw[:100],
        y_test=y_test[:100],
    )
    print(f"  已保存：{vis_path}（原始图像前 500/100 张用于可视化）")

    # ------------------------------------------------------------------
    # 4. 统计摘要
    # ------------------------------------------------------------------
    print("\n[4/4] 数据统计摘要")
    class_names = get_class_names()
    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)
    print(f"  {'类别':<12} {'训练集数量':>10} {'测试集数量':>10}")
    print(f"  {'-'*34}")
    for i, name in enumerate(class_names):
        print(f"  {name:<12} {train_counts[i]:>10} {test_counts[i]:>10}")

    print("\n[步骤一] 完成！\n")


if __name__ == "__main__":
    main()
