import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


if __name__ == '__main__':
    file_list = ['road_mark_rtdetr.csv','road_mark_v3.csv', 'road_mark_v5.csv', 'road_mark_v6.csv', 'road_mark_v8.csv', 'road_mark_v9.csv', 'road_mark_v10.csv', 'road_mark_12.csv', 'road_mark_11.csv', 'road_mark_11_CGA.csv']
    # file_list = ['brain_tumor_02_rtdetr.csv','v3.csv', 'v5.csv', 'v6.csv', 'v8n.csv', 'v9.csv', 'v10.csv', 'brain_tumor_12.csv', '11n.csv', '11n-C2CGA.csv']
    names = ['RT-DETR', 'YOLOv3', 'YOLOv5', 'YOLOv6', 'YOLOv8', 'YOLOv9', 'YOLOv10', 'YOLO12', 'YOLO11(baseline)', 'YOLO11-C2CGA']

    plt.figure(figsize=(6, 6))
    for i in range(len(file_list)):
        pr_data = pd.read_csv(file_list[i], header=None)
        recall, precision = np.array(pr_data[0]), np.array(pr_data[1])

        # 平滑处理
        recall_smooth = smooth(recall)
        precision_smooth = smooth(precision)

        if 'C2CGA' in str(names[i]):
            plt.plot(recall_smooth, precision_smooth, linewidth=3, color="blue", label=f'{names[i]}')
        else:
            plt.plot(recall_smooth, precision_smooth, linewidth=1, label=f'{names[i]}')

        # 添加网格线
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('PR_curve_brain_tumor-3.svg')
    plt.savefig('PR_curve_road_mark-3.svg')
