import matplotlib.pyplot as plt
import numpy as np

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16

# Data
learning_rates = [0.001, 0.005, 0.01, 0.03, 0.05]
models = ['YOLO11', 'YOLO11-C2CGA']

precision_yolo11 = [76.5, 84.9, 87.8, 86.1, 88.4]
# precision_yolo11 = [86.4, 91.2, 92.0, 92.6, 92.1]
precision_yolo11_c2cga = [66.8, 89.5, 89.4, 88.3, 86.3]
# precision_yolo11_c2cga = [90.3, 93.5, 91.3, 93.5, 91.0]

recall_yolo11 = [72.4, 84.8, 86.5, 83.6, 85.5]
# recall_yolo11 = [88.3, 90.3, 90.3, 91.1, 90.2]
recall_yolo11_c2cga = [72.1, 83.1, 87.2, 85.8, 83.9]
# recall_yolo11_c2cga = [86.7, 89.6, 91.6, 88.8, 90.2]

mAP_yolo11 = [77.2, 88.0, 88.4, 87.8, 88.4]
# mAP_yolo11 = [90.3, 93.0, 92.6, 93.7, 92.2]
mAP_yolo11_c2cga = [72.7, 87.5, 90.6, 90.0, 89.1]
# mAP_yolo11_c2cga = [90.6, 92.6, 94.1, 92.6, 92.8]

f1_score_yolo11 = [74.4, 84.8, 87.1, 84.8, 86.9]
# f1_score_yolo11 = [87.3, 90.7, 91.1, 92.4, 91.2]
f1_score_yolo11_c2cga = [69.3, 86.1, 88.3, 87.0, 85.1]
# f1_score_yolo11_c2cga = [88.5, 91.5, 91.4, 91.1, 90.6]

fps_yolo11 = [179.1, 186.7, 208.5, 202.0, 199.6]
# fps_yolo11 = [153.7, 210.1, 204.3, 193.1, 186.8]
fps_yolo_c2cga =[175.4, 176.6, 181.1, 183.2, 192.7]
# fps_yolo_c2cga =[170.1, 190.1, 183.1, 195.6, 195.3]

# Set colors
color1 = '#1f77b4'  # Blue
color2 = '#ff7f0e'  # Orange

# Create subplots in a single row
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

# Bar width
bar_width = 0.3
x = np.arange(len(learning_rates))

# Plot each metric
metrics = [
    ('Precision', precision_yolo11, precision_yolo11_c2cga),
    ('Recall', recall_yolo11, recall_yolo11_c2cga),
    ('mAP', mAP_yolo11, mAP_yolo11_c2cga),
    ('F1-Score', f1_score_yolo11, f1_score_yolo11_c2cga),
    ('FPS', fps_yolo11, fps_yolo_c2cga)
]

for ax, (metric_name, y1, y2) in zip(axes, metrics):
    bar1 = ax.bar(x - bar_width / 2, y1, width=bar_width, label='YOLO11', color=color1)
    bar2 = ax.bar(x + bar_width / 2, y2, width=bar_width, label='YOLO11-C2CGA', color=color2)

    # Annotate bars
    for i, v in enumerate(y1):
        ax.text(x[i] - bar_width / 2, v + 0.5, str(v), ha='center', va='bottom', fontsize=8 if metric_name != 'FPS' else 7)
    for i, v in enumerate(y2):
        ax.text(x[i] + bar_width / 2, v + 0.5, str(v), ha='center', va='bottom', fontsize=8 if metric_name != 'FPS' else 7)

    ax.set_xlabel('Learning Rate')
    ax.set_ylabel(f'{metric_name} (%)' if metric_name != 'FPS' else 'Speed (FPS)')
    ax.set_xticks(x)
    ax.set_xticklabels(learning_rates)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--')
    ax.set_ylim(0, 112 if metric_name != 'FPS' else 255)

# Adjust layout
plt.tight_layout()

# Save as PNG file
plt.savefig('learning_rate_road_mark.pdf')

# Show the plot
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import MaxNLocator
#
# # 设置全局字体为 Times New Roman
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['font.size'] = 12
#
# # 数据
# learning_rates = [0.001, 0.005, 0.01, 0.03, 0.05]
# models = ['YOLO11', 'YOLO11+C2CGA']
#
# # 各个指标数据
# precision_yolo11 = [76.5, 84.9, 87.8, 86.1, 88.4]
# # precision_yolo11 = [86.4, 91.2, 92.0, 92.6, 92.1]
# precision_yolo11_c2cga = [66.8, 89.5, 89.4, 88.3, 86.3]
# # precision_yolo11_c2cga = [90.3, 93.5, 91.3, 93.5, 91.0]
#
# recall_yolo11 = [72.4, 84.8, 86.5, 83.6, 85.5]
# # recall_yolo11 = [88.3, 90.3, 90.3, 91.1, 90.2]
# recall_yolo11_c2cga = [72.1, 83.1, 87.2, 85.8, 83.9]
# # recall_yolo11_c2cga = [86.7, 89.6, 91.6, 88.8, 90.2]
#
# mAP_yolo11 = [77.2, 88.0, 88.4, 87.8, 88.4]
# # mAP_yolo11 = [90.3, 93.0, 92.6, 93.7, 92.2]
# mAP_yolo11_c2cga = [72.7, 87.5, 90.6, 90.0, 89.1]
# # mAP_yolo11_c2cga = [90.6, 92.6, 94.1, 92.6, 92.8]
#
# f1_score_yolo11 = [74.4, 84.8, 87.1, 84.8, 86.9]
# # f1_score_yolo11 = [87.3, 90.7, 91.1, 92.4, 91.2]
# f1_score_yolo11_c2cga = [69.3, 86.1, 88.3, 87.0, 85.1]
# # f1_score_yolo11_c2cga = [88.5, 91.5, 91.4, 91.1, 90.6]
#
# fps_yolo11 = [179.1, 186.7, 208.5, 202.0, 199.6]
# # fps_yolo11 = [153.7, 210.1, 204.3, 193.1, 186.8]
# fps_yolo_c2cga =[175.4, 176.6, 181.1, 183.2, 192.7]
# # fps_yolo_c2cga =[170.1, 190.1, 183.1, 195.6, 195.3]
#
# # 设置颜色
# color1 = '#1f77b4'  # 蓝色
# color2 = '#ff7f0e'  # 橙色
#
# # 创建子图并分别绘制四个指标
# fig, axes = plt.subplots(1, 5, figsize=(15, 10))
#
# # 绘制precision图表
# ax1 = axes[0, 0]
# bar_width = 0.3
# x1 = np.array(range(len(learning_rates)))
#
# # 绘制YOLO11的precision柱状图
# ax1.bar(x1 - bar_width/2, precision_yolo11, width=bar_width, label='YOLO11', color=color1)
# # 绘制YOLO11+C2CGA的precision柱状图
# ax1.bar(x1 + bar_width/2, precision_yolo11_c2cga, width=bar_width, label='YOLO11+C2CGA', color=color2)
#
# # 在每个柱子上标注数值
# for i, v in enumerate(precision_yolo11):
#     ax1.text(x1[i] - bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
# for i, v in enumerate(precision_yolo11_c2cga):
#     ax1.text(x1[i] + bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
#
# ax1.set_xlabel('Learning Rate')
# ax1.set_ylabel('Precision (%)')
# ax1.set_xticks(x1)
# ax1.set_xticklabels(learning_rates)
# ax1.legend()
# ax1.grid(True, linestyle='--')
#
# # 手动设置Y轴范围
# ax1.set_ylim(0, 115)
#
# # 绘制recall图表
# ax2 = axes[0, 1]
# x2 = np.array(range(len(learning_rates)))
#
# # 绘制YOLO11的recall柱状图
# ax2.bar(x2 - bar_width/2, recall_yolo11, width=bar_width, label='YOLO11', color=color1)
# # 绘制YOLO11+C2CGA的recall柱状图
# ax2.bar(x2 + bar_width/2, recall_yolo11_c2cga, width=bar_width, label='YOLO11+C2CGA', color=color2)
#
# # 在每个柱子上标注数值
# for i, v in enumerate(recall_yolo11):
#     ax2.text(x2[i] - bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
# for i, v in enumerate(recall_yolo11_c2cga):
#     ax2.text(x2[i] + bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
#
# ax2.set_xlabel('Learning Rate')
# ax2.set_ylabel('Recall (%)')
# ax2.set_xticks(x2)
# ax2.set_xticklabels(learning_rates)
# ax2.legend()
# ax2.grid(True, linestyle='--')
#
# # 手动设置Y轴范围
# ax2.set_ylim(0, 115)
#
# # 绘制mAP图表
# ax3 = axes[0, 2]
# x3 = np.array(range(len(learning_rates)))
#
# # 绘制YOLO11的mAP柱状图
# ax3.bar(x3 - bar_width/2, mAP_yolo11, width=bar_width, label='YOLO11', color=color1)
# # 绘制YOLO11+C2CGA的mAP柱状图
# ax3.bar(x3 + bar_width/2, mAP_yolo11_c2cga, width=bar_width, label='YOLO11+C2CGA', color=color2)
#
# # 在每个柱子上标注数值
# for i, v in enumerate(mAP_yolo11):
#     ax3.text(x3[i] - bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
# for i, v in enumerate(mAP_yolo11_c2cga):
#     ax3.text(x3[i] + bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
#
# ax3.set_xlabel('Learning Rate')
# ax3.set_ylabel('mAP (%)')
# ax3.set_xticks(x3)
# ax3.set_xticklabels(learning_rates)
# ax3.legend()
# ax3.grid(True, linestyle='--')
#
# # 手动设置Y轴范围
# ax3.set_ylim(0, 115)
#
# # 绘制f1-score图表
# ax4 = axes[0, 3]
# x4 = np.array(range(len(learning_rates)))
#
# # 绘制YOLO11的f1-score柱状图
# ax4.bar(x4 - bar_width/2, f1_score_yolo11, width=bar_width, label='YOLO11', color=color1)
# # 绘制YOLO11+C2CGA的f1-score柱状图
# ax4.bar(x4 + bar_width/2, f1_score_yolo11_c2cga, width=bar_width, label='YOLO11+C2CGA', color=color2)
#
# # 在每个柱子上标注数值
# for i, v in enumerate(f1_score_yolo11):
#     ax4.text(x4[i] - bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
# for i, v in enumerate(f1_score_yolo11_c2cga):
#     ax4.text(x4[i] + bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
#
# ax4.set_xlabel('Learning Rate')
# ax4.set_ylabel('F1-Score (%)')
# ax4.set_xticks(x4)
# ax4.set_xticklabels(learning_rates)
# ax4.legend()
# ax4.grid(True, linestyle='--')
#
# # 手动设置Y轴范围
# ax4.set_ylim(0, 115)
#
# ax5 = axes[0, 4]
# x5 = np.array(range(len(learning_rates)))
# ax5.bar(x5 - bar_width/2, fps_yolo11, width=bar_width, label='YOLO11', color=color1)
# ax5.bar(x5 + bar_width/2, fps_yolo_c2cga, width=bar_width, label='YOLO11+C2CGA', color=color2)
#
# for i, v in enumerate(fps_yolo11):
#     ax5.text(x5[i] - bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=7)
# for i, v in enumerate(fps_yolo_c2cga):
#     ax5.text(x5[i] + bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=7)
#
# ax5.set_xlabel('Learning Rate')
# ax5.set_ylabel('Speed (FPS)')
# ax5.set_xticks(x5)
# ax5.set_xticklabels(learning_rates)
# ax5.legend()
# ax5.grid(True, linestyle='--')
# ax5.set_ylim(0, 250)
#
#
# # 自动调整布局
# plt.tight_layout()
#
# # 保存为pdf文件
# plt.savefig('learning_rate_road_mark_2.png')
#
# # 显示图表
# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import MaxNLocator
#
# # 设置全局字体为 Times New Roman
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['font.size'] = 12
#
# # 数据
# learning_rates = [0.001, 0.005, 0.01, 0.03, 0.05]
# models = ['YOLO11', 'YOLO11+C2CGA']
#
# # 各个指标数据
# # precision_yolo11 = [76.5, 84.9, 87.8, 86.1, 88.4]
# precision_yolo11 = [86.4, 91.2, 92.0, 92.6, 92.1]
# # precision_yolo11_c2cga = [66.8, 89.5, 89.4, 88.3, 86.3]
# precision_yolo11_c2cga = [90.3, 93.5, 91.3, 93.5, 91.0]
#
# # recall_yolo11 = [72.4, 84.8, 86.5, 83.6, 85.5]
# recall_yolo11 = [88.3, 90.3, 90.3, 91.1, 90.2]
# # recall_yolo11_c2cga = [72.1, 83.1, 87.2, 85.8, 83.9]
# recall_yolo11_c2cga = [86.7, 89.6, 91.6, 88.8, 90.2]
#
# # mAP_yolo11 = [77.2, 88.0, 88.4, 87.8, 88.4]
# mAP_yolo11 = [90.3, 93.0, 92.6, 93.7, 92.2]
# # mAP_yolo11_c2cga = [72.7, 87.5, 90.6, 90.0, 89.1]
# mAP_yolo11_c2cga = [90.6, 92.6, 94.1, 92.6, 92.8]
#
# # f1_score_yolo11 = [74.4, 84.8, 87.1, 84.8, 86.9]
# f1_score_yolo11 = [87.3, 90.7, 91.1, 92.4, 91.2]
# # f1_score_yolo11_c2cga = [69.3, 86.1, 88.3, 87.0, 85.1]
# f1_score_yolo11_c2cga = [88.5, 91.5, 91.4, 91.1, 90.6]
#
# # 设置颜色
# color1 = '#1f77b4'  # 蓝色
# color2 = '#ff7f0e'  # 橙色
#
# # 创建子图并分别绘制四个指标
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#
# # 绘制precision图表
# ax1 = axes[0, 0]
# bar_width = 0.3
# x1 = np.array(range(len(learning_rates)))
#
# # 绘制YOLO11的precision柱状图
# ax1.bar(x1 - bar_width/2, precision_yolo11, width=bar_width, label='YOLO11', color=color1)
# # 绘制YOLO11+C2CGA的precision柱状图
# ax1.bar(x1 + bar_width/2, precision_yolo11_c2cga, width=bar_width, label='YOLO11+C2CGA', color=color2)
#
# # 在每个柱子上标注数值
# for i, v in enumerate(precision_yolo11):
#     ax1.text(x1[i] - bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
# for i, v in enumerate(precision_yolo11_c2cga):
#     ax1.text(x1[i] + bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
#
# ax1.set_xlabel('Learning Rate')
# ax1.set_ylabel('Precision (%)')
# ax1.set_xticks(x1)
# ax1.set_xticklabels(learning_rates)
# ax1.legend()
# ax1.grid(True, linestyle='--')
#
# # 手动设置Y轴范围
# ax1.set_ylim(0, 115)
#
# # 绘制recall图表
# ax2 = axes[0, 1]
# x2 = np.array(range(len(learning_rates)))
#
# # 绘制YOLO11的recall柱状图
# ax2.bar(x2 - bar_width/2, recall_yolo11, width=bar_width, label='YOLO11', color=color1)
# # 绘制YOLO11+C2CGA的recall柱状图
# ax2.bar(x2 + bar_width/2, recall_yolo11_c2cga, width=bar_width, label='YOLO11+C2CGA', color=color2)
#
# # 在每个柱子上标注数值
# for i, v in enumerate(recall_yolo11):
#     ax2.text(x2[i] - bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
# for i, v in enumerate(recall_yolo11_c2cga):
#     ax2.text(x2[i] + bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
#
# ax2.set_xlabel('Learning Rate')
# ax2.set_ylabel('Recall (%)')
# ax2.set_xticks(x2)
# ax2.set_xticklabels(learning_rates)
# ax2.legend()
# ax2.grid(True, linestyle='--')
#
# # 手动设置Y轴范围
# ax2.set_ylim(0, 115)
#
# # 绘制mAP图表
# ax3 = axes[1, 0]
# x3 = np.array(range(len(learning_rates)))
#
# # 绘制YOLO11的mAP柱状图
# ax3.bar(x3 - bar_width/2, mAP_yolo11, width=bar_width, label='YOLO11', color=color1)
# # 绘制YOLO11+C2CGA的mAP柱状图
# ax3.bar(x3 + bar_width/2, mAP_yolo11_c2cga, width=bar_width, label='YOLO11+C2CGA', color=color2)
#
# # 在每个柱子上标注数值
# for i, v in enumerate(mAP_yolo11):
#     ax3.text(x3[i] - bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
# for i, v in enumerate(mAP_yolo11_c2cga):
#     ax3.text(x3[i] + bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
#
# ax3.set_xlabel('Learning Rate')
# ax3.set_ylabel('mAP (%)')
# ax3.set_xticks(x3)
# ax3.set_xticklabels(learning_rates)
# ax3.legend()
# ax3.grid(True, linestyle='--')
#
# # 手动设置Y轴范围
# ax3.set_ylim(0, 115)
#
# # 绘制f1-score图表
# ax4 = axes[1, 1]
# x4 = np.array(range(len(learning_rates)))
#
# # 绘制YOLO11的f1-score柱状图
# ax4.bar(x4 - bar_width/2, f1_score_yolo11, width=bar_width, label='YOLO11', color=color1)
# # 绘制YOLO11+C2CGA的f1-score柱状图
# ax4.bar(x4 + bar_width/2, f1_score_yolo11_c2cga, width=bar_width, label='YOLO11+C2CGA', color=color2)
#
# # 在每个柱子上标注数值
# for i, v in enumerate(f1_score_yolo11):
#     ax4.text(x4[i] - bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
# for i, v in enumerate(f1_score_yolo11_c2cga):
#     ax4.text(x4[i] + bar_width/2, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
#
# ax4.set_xlabel('Learning Rate')
# ax4.set_ylabel('F1-Score (%)')
# ax4.set_xticks(x4)
# ax4.set_xticklabels(learning_rates)
# ax4.legend()
# ax4.grid(True, linestyle='--')
#
# # 手动设置Y轴范围
# ax4.set_ylim(0, 115)
#
# # 自动调整布局
# plt.tight_layout()
#
# # 保存为pdf文件
# plt.savefig('learning_rate_brain_tumor.pdf')
#
# # 显示图表
# plt.show()
