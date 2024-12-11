# import matplotlib.pyplot as plt
# import numpy as np

# # 方法的名称列表
# methods = ['EEGNet', 'DAN', 'DANN', 'JAN', 'CDAN', 'MDD', 'MCC', 'SHOT', 'ISFDA', 'MAD(Ours)']

# # 假设有9个被试和10种方法
# num_methods = 10  # 横坐标表示的10种方法
# num_subjects = 8  # 9个被试

# # 9个被试的Acc数据，保留小数点后两位
# acc_data = np.array([
#     [74.45, 75.21, 74.46, 75.85, 76.04, 74.93, 76.75, 74.92, 58.30, 77.90],
#     [66.55, 67.40, 66.06, 68.90, 69.41, 66.34, 69.56, 66.71, 52.77, 72.20],
#     [79.23, 79.42, 79.95, 79.85, 80.43, 79.69, 80.82, 79.53, 59.08, 81.04],
#     [67.46, 67.79, 67.87, 68.48, 68.53, 67.58, 69.31, 70.77, 55.14, 71.79],
#     [68.48, 68.93, 68.54, 69.60, 70.65, 69.15, 74.95, 72.85, 71.21, 73.84],
#     [69.78, 71.80, 70.48, 71.91, 73.74, 71.07, 74.59, 72.49, 54.28, 77.20],
#     [68.68, 70.00, 69.16, 71.42, 72.40, 69.17, 72.89, 72.33, 61.48, 74.65],
#     [77.05, 77.85, 77.19, 80.13, 81.53, 76.29, 86.23, 83.65, 71.76, 85.01]
# ])

# # 计算每种方法的平均Acc
# mean_acc = np.mean(acc_data, axis=0)
# print(mean_acc)

# # 根据图中的SCI配色设置颜色
# subject_colors = [
#     '#4E8F89', '#EEDF70', '#E9D26F', '#D97943', '#824028', '#FF4500', '#FFD700', '#87CEEB'
# ]
# mean_color = '#FF0000'  # 亮红色表示平均值

# # 创建图形
# plt.figure(figsize=(12, 8))

# # 为每个被试绘制一条折线，使用自定义的配色
# for subject in range(num_subjects):
#     plt.plot(methods, acc_data[subject], marker='o', label=f'Subject {subject+1}', color=subject_colors[subject], linewidth=2, alpha=0.8)

# # 绘制平均值折线，使用亮红色
# plt.plot(methods, mean_acc, marker='o', label='Mean Acc', linewidth=6, color=mean_color)

# # 设置图形属性
# plt.ylim(50, 100)  # 纵坐标范围设置为60-100
# plt.xlabel('Methods', fontsize=14)
# plt.ylabel('Accuracy (Acc)', fontsize=14)
# plt.title('Acc of Different Subjects and Mean Using 10 Methods', fontsize=16)

# # 显示图例
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 通过bbox_to_anchor调整图例位置

# plt.tight_layout()
# plt.savefig('./009_008_uda.pdf')
# plt.show()





import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 设置所有字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 方法的名称列表
methods = ['EEGNet', 'DAN', 'DANN', 'JAN', 'CDAN', 'MDD', 'MCC', 'SHOT', 'ISFDA', 'SDDA(Ours)']

# 假设有9个被试和10种方法
num_methods = 10  # 横坐标表示的10种方法
num_subjects = 8  # 9个被试

# 9个被试的Acc数据，保留小数点后两位
acc_data = np.array([
    [74.45, 75.21, 74.46, 75.85, 76.04, 74.93, 76.75, 74.92, 58.30, 77.90],
    [66.55, 67.40, 66.06, 68.90, 69.41, 66.34, 69.56, 66.71, 52.77, 72.20],
    [79.23, 79.42, 79.95, 79.85, 80.43, 79.69, 80.82, 79.53, 59.08, 81.04],
    [67.46, 67.79, 67.87, 68.48, 68.53, 67.58, 69.31, 70.77, 55.14, 71.79],
    [68.48, 68.93, 68.54, 69.60, 70.65, 69.15, 74.95, 72.85, 71.21, 73.84],
    [69.78, 71.80, 70.48, 71.91, 73.74, 71.07, 74.59, 72.49, 54.28, 77.20],
    [68.68, 70.00, 69.16, 71.42, 72.40, 69.17, 72.89, 72.33, 61.48, 74.65],
    [77.05, 77.85, 77.19, 80.13, 81.53, 76.29, 86.23, 83.65, 71.76, 85.01]
])

# 计算每种方法的平均Acc
mean_acc = np.mean(acc_data, axis=0)
print(mean_acc)

# subject_colors to match your customization
subject_colors = ['#B42B22', '#2B4871', '#996E2E', '#E995C9', '#1E90FF', '#F6944B', '#009688', '#830783']
mean_color = '#FF0000'  # Red for mean accuracy

# 创建一个包含两个子图的图形（一个用于显示平均值，一个用于显示各个被试的Acc）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

# 在第一个子图中绘制平均Acc曲线
marker_style = 'D'
ax1.plot(methods, mean_acc, marker=marker_style, label='Mean AUC', linewidth=4, alpha=0.5, color=mean_color)
ax1.set_ylim(60, 78)  # 调整纵坐标范围以获得更好的显示效果
ax1.set_ylabel('Mean AUC (%)', fontsize=22, labelpad=10)

# 在第一个子图上为最后一列添加灰色背景
ax1.add_patch(Rectangle((9 - 0.4, 60), 0.8, 18, color='gray', alpha=0.2))

# 设置Y轴刻度为68, 71, 74
ax1.set_yticks([65, 70, 75])

# 调整刻度字体大小
ax1.tick_params(axis='both', which='major', labelsize=22)

# 只在这几个主要刻度上绘制网格线
ax1.grid(True, which='major', axis='y', color='gray', linestyle='--', linewidth=0.7)

# 将图例移到左上角，使用弧形框框并添加较深的灰色背景，边框为深色
# ax1.legend(loc='upper left', fancybox=True, framealpha=0.5, facecolor='#e0e0e0', edgecolor='black', fontsize=14)

# 设置标题
ax1.set_title('Offline: BNCI2014009→BNCI2014008', fontsize=22, fontweight='bold')

# 将横坐标标签倾斜30°
plt.xticks(rotation=30, fontsize=22)
ax1.set_yticklabels(ax1.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型




# 在第二个子图中绘制各个被试的Acc曲线
for subject in range(num_subjects):
    ax2.plot(methods, acc_data[subject], marker=marker_style, label=f'Subject {subject + 1}', color=subject_colors[subject], linewidth=2)

# 为第二个子图的最后一列添加灰色背景
ax2.add_patch(Rectangle((9 - 0.4, 52), 0.8, 36, color='gray', alpha=0.2))

# 设置第二个子图的属性
ax2.set_ylim(52, 88)  # 纵坐标范围设置为54-95
# ax2.set_xlabel('Approaches', fontsize=18)
ax2.set_yticks(np.arange(55, 86, 5))  # 设置刻度步长为5，确保是整数
ax2.set_ylabel('Area Under the Curve (%)', fontsize=22, labelpad=10)

# 调整刻度字体大小
ax2.tick_params(axis='both', which='major', labelsize=22)

# 添加Y轴的网格线
ax2.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)

# 显示图例，使用弧形框框并添加较深的灰色背景，边框为深色
# ax2.legend(loc='upper left', fancybox=True, framealpha=0.5, facecolor='#e0e0e0', edgecolor='black', ncol=2, fontsize=22)

# 将横坐标标签倾斜30°
plt.xticks(rotation=30, fontsize=22, fontweight='bold')
ax1.set_yticklabels(ax1.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 x 轴刻度字体大小和字体类型
ax2.set_yticklabels(ax2.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型

# 调整布局并保存图像
plt.tight_layout()
plt.savefig('./009_008_uda.pdf')
plt.show()
