import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')

def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

# 预先加载数据到字典
def load_and_process_data(names):
    data_dict = {}
    for name in names:
        data = pd.read_csv(f'runs/train/{name}/results.csv')
        data.columns = data.columns.str.strip()

        cols_to_clean = [
            'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'train/box_loss', 'train/dfl_loss', 'train/cls_loss',
            'val/box_loss', 'val/dfl_loss', 'val/cls_loss'
        ]

        for col in cols_to_clean:
            if col in data.columns:
                data[col] = data[col].astype(np.float32).replace([np.inf, -np.inf], np.nan)
                data[col] = data[col].interpolate(method='polynomial', order=2).fillna(method='ffill').fillna(method='bfill')

        data_dict[name] = data
    return data_dict

pwd = os.getcwd()
# names = ['brain_tumor_02_11', 'brain_tumor_02_11_CGA']
# labels = ['YOLO11', 'YOLO11-C2CGA']
names = ['brain_tumor_02_rtdetr','v3', 'v5', 'v6', 'v8', 'v9','v10', '12', 'brain_tumor_02_11', 'brain_tumor_02_11_CGA']
# names = ['road_mark_rtdetr','road_mark_v3', 'road_mark_v5', 'road_mark_v6', 'road_mark_v8', 'road_mark_v9','road_mark_v10', 'road_mark_12', 'road_mark_11', 'road_mark_11_CGA']
labels = ['RT-DETR','YOLOv3', 'YOLOv5', 'YOLOv6', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLO12', 'YOLO11(baseline)', 'YOLO11-C2CGA']

data_dict = load_and_process_data(names)

# 定义绘图函数（使用自定义平滑）
def plot_smoothed(data_dict, column, title, subplot_position, f=0.05):
    plt.subplot(subplot_position)
    for i, name in enumerate(names):
        data = data_dict[name]
        if column in data.columns:
            # 使用自定义平滑函数
            smoothed = smooth(data[column].values, f=f)
            if 'CGA' in name:
                plt.plot(smoothed, linewidth=3, color="blue", label=labels[i])
            else:
                plt.plot(smoothed, label=labels[i])
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.legend()

    # 添加网格线
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# 绘制指标图
plt.figure(figsize=(15, 5))
plot_smoothed(data_dict, 'metrics/precision(B)', 'Precision', 131)
plot_smoothed(data_dict, 'metrics/recall(B)', 'Recall', 132)
plot_smoothed(data_dict, 'metrics/mAP50(B)', 'mAP@0.5', 133)

plt.tight_layout()
plt.savefig('brain_tumor_metrics_curve_2.pdf')


