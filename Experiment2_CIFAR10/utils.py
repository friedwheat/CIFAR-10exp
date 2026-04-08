"""
utils.py – 通用工具函数（加载数据、预处理、转换逻辑）

CIFAR-10 原始二进制批次文件格式说明：
  每个批次文件包含 10 000 张 32×32 的彩色图像，以 pickle 格式存储。
  每张图像展开为长度 3072 的向量（1024 R + 1024 G + 1024 B）。
"""

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np

# CIFAR-10 类别名称（按官方标签索引排列）
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


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_cifar10_batch(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载单个 CIFAR-10 批次文件。

    Parameters
    ----------
    filepath : str
        批次文件的完整路径（如 ``data/data_batch_1``）。

    Returns
    -------
    X : np.ndarray, shape (N, 3072), dtype float32
        原始像素值，范围 [0, 255]。
    y : np.ndarray, shape (N,), dtype int64
        类别标签，取值 0–9。
    """
    with open(filepath, "rb") as f:
        entry = pickle.load(f, encoding="bytes")
    X = entry[b"data"].astype(np.float32)
    y = np.array(entry[b"labels"], dtype=np.int64)
    return X, y


def load_cifar10(
    data_dir: str,
    num_train_batches: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载完整的 CIFAR-10 训练集与测试集。

    Parameters
    ----------
    data_dir : str
        存放批次文件的目录路径。
    num_train_batches : int
        要加载的训练批次数（最多 5）。

    Returns
    -------
    X_train : np.ndarray, shape (N_train, 3072)
    y_train : np.ndarray, shape (N_train,)
    X_test  : np.ndarray, shape (N_test, 3072)
    y_test  : np.ndarray, shape (N_test,)
    """
    X_list, y_list = [], []
    for i in range(1, num_train_batches + 1):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X, y = load_cifar10_batch(batch_path)
        X_list.append(X)
        y_list.append(y)
    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)

    test_path = os.path.join(data_dir, "test_batch")
    X_test, y_test = load_cifar10_batch(test_path)

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------

def normalize_pixels(X: np.ndarray) -> np.ndarray:
    """将像素值从 [0, 255] 缩放到 [0, 1]。"""
    return X / 255.0


def standardize(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """零均值、单位方差标准化。

    Parameters
    ----------
    X    : np.ndarray, shape (N, D)
    mean : 可选，外部提供的均值（例如训练集均值）。
    std  : 可选，外部提供的标准差。

    Returns
    -------
    X_std : 标准化后的数据
    mean  : 使用的均值
    std   : 使用的标准差
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # 避免除以 0：以 1.0 替代零标准差，使该维度标准化后保持原值
    return (X - mean) / std, mean, std


# ---------------------------------------------------------------------------
# 形状转换
# ---------------------------------------------------------------------------

def reshape_to_chw(X: np.ndarray) -> np.ndarray:
    """将 (N, 3072) 转为 (N, 3, 32, 32)（通道优先）。"""
    return X.reshape(-1, 3, 32, 32)


def reshape_to_hwc(X: np.ndarray) -> np.ndarray:
    """将 (N, 3072) 转为 (N, 32, 32, 3)（通道最后），便于 matplotlib 显示。"""
    return X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def to_uint8(X_hwc: np.ndarray) -> np.ndarray:
    """将浮点图像 [0, 1] 转换为 uint8 [0, 255]，用于显示。"""
    return np.clip(X_hwc * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

def extract_flatten(X: np.ndarray) -> np.ndarray:
    """直接展平像素作为特征（已归一化到 [0, 1]）。

    Parameters
    ----------
    X : np.ndarray, shape (N, 3072)，像素值范围 [0, 255]

    Returns
    -------
    np.ndarray, shape (N, 3072)，值域 [0, 1]
    """
    return normalize_pixels(X)


def extract_grayscale_flatten(X: np.ndarray) -> np.ndarray:
    """将 RGB 图像转为灰度后展平。

    Parameters
    ----------
    X : np.ndarray, shape (N, 3072)，像素值范围 [0, 255]

    Returns
    -------
    np.ndarray, shape (N, 1024)，灰度值域 [0, 1]
    """
    X_norm = normalize_pixels(X)
    X_hwc = X_norm.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    gray = (
        0.2989 * X_hwc[:, :, :, 0]
        + 0.5870 * X_hwc[:, :, :, 1]
        + 0.1140 * X_hwc[:, :, :, 2]
    )
    return gray.reshape(len(X), -1)


def extract_color_histogram(X: np.ndarray, bins: int = 32) -> np.ndarray:
    """为每张图像提取 RGB 三通道的颜色直方图并拼接。

    Parameters
    ----------
    X    : np.ndarray, shape (N, 3072)，像素值范围 [0, 255]
    bins : 每个通道的直方图 bin 数量

    Returns
    -------
    np.ndarray, shape (N, 3 * bins)，归一化直方图特征
    """
    N = X.shape[0]
    X_chw = X.reshape(N, 3, 32, 32)
    features = np.empty((N, 3 * bins), dtype=np.float32)
    for i in range(N):
        for c in range(3):
            hist, _ = np.histogram(X_chw[i, c], bins=bins, range=(0.0, 255.0))
            features[i, c * bins : (c + 1) * bins] = hist.astype(np.float32)
    # L1 归一化
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return features / row_sums


# ---------------------------------------------------------------------------
# 杂项
# ---------------------------------------------------------------------------

def get_class_names() -> List[str]:
    """返回 CIFAR-10 类别名称列表（共 10 类）。"""
    return CIFAR10_CLASSES.copy()


def class_means(X: np.ndarray, y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """计算每个类别在特征空间中的均值向量。

    Parameters
    ----------
    X           : np.ndarray, shape (N, D)
    y           : np.ndarray, shape (N,)
    num_classes : int

    Returns
    -------
    means : np.ndarray, shape (num_classes, D)
    """
    D = X.shape[1]
    means = np.zeros((num_classes, D), dtype=np.float64)
    for c in range(num_classes):
        mask = y == c
        if mask.sum() > 0:
            means[c] = X[mask].mean(axis=0)
    return means
