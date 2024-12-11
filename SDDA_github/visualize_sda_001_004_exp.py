# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Rectangle

# # 设置所有字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'

# # 假设有9个被试和11种方法
# num_methods = 11  # 横坐标表示的11种方法
# num_subjects = 9  # 9个被试

# # 方法的名称列表
# methods = ['CSP+LDA', 'EEGNet', 'DAN', 'DANN', 'JAN', 'CDAN', 'MDD', 'MCC', 'SHOT', 'ISFDA', 'MAD(Ours)']

# # 9个被试的Acc数据，保留小数点后两位，增加第11种方法的数据
# acc_data = np.array([
#     [63.66, 66.34, 66.48, 65.29, 66.98, 66.80, 67.50, 67.09, 65.09, 60.55, 70.73],
#     [56.17, 53.61, 53.92, 55.32, 54.51, 54.63, 54.85, 55.09, 56.17, 54.72, 56.02],
#     [54.94, 56.77, 57.15, 55.81, 56.54, 56.89, 55.67, 55.99, 56.40, 58.08, 57.09],
#     [88.42, 89.97, 90.34, 89.83, 88.33, 89.83, 92.60, 92.83, 86.24, 87.83, 93.96],
#     [75.28, 73.39, 72.85, 74.83, 74.58, 75.09, 75.28, 73.31, 74.38, 72.03, 78.25],
#     [75.00, 71.40, 71.74, 70.81, 70.23, 71.42, 71.34, 70.64, 72.21, 69.42, 74.65],
#     [68.75, 70.00, 71.80, 67.09, 71.40, 72.91, 70.96, 72.21, 71.42, 67.65, 72.53],
#     [77.89, 76.54, 77.47, 77.23, 76.81, 76.48, 77.01, 77.25, 76.95, 75.93, 79.45],
#     [74.86, 70.29, 72.18, 72.04, 70.20, 72.06, 71.19, 70.35, 68.66, 67.06, 75.44]
# ])

# # 计算每种方法的平均Acc
# mean_acc = np.mean(acc_data, axis=0)

# # subject_colors to match your customization
# subject_colors = ['#B42B22', '#2B4871', '#996E2E', '#E995C9', '#1E90FF', '#F6944B', '#7E4909', '#009688', '#830783']
# mean_color = '#FF0000'  # Red for mean accuracy

# # 创建一个包含两个子图的图形（一个用于显示平均值，一个用于显示各个被试的Acc）
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

# # 在第一个子图中绘制平均Acc曲线
# marker_style = 'D'
# ax1.plot(methods, mean_acc, marker=marker_style, label='Mean Acc', linewidth=4, alpha=0.5, color=mean_color)
# ax1.set_ylim(67, 74)  # 调整纵坐标范围以获得更好的显示效果
# ax1.set_ylabel('Mean Accuracy (%)', fontsize=19, labelpad=10)

# # 在第一个子图上为最后一列添加灰色背景
# ax1.add_patch(Rectangle((10 - 0.4, 67), 0.8, 7, color='gray', alpha=0.2))

# # 设置Y轴刻度为68, 71, 74
# ax1.set_yticks([67, 70, 73])

# # 调整刻度字体大小
# ax1.tick_params(axis='both', which='major', labelsize=14)

# # 只在这几个主要刻度上绘制网格线
# ax1.grid(True, which='major', axis='y', color='gray', linestyle='--', linewidth=0.7)

# # 将图例移到左上角，使用弧形框框并添加较深的灰色背景，边框为深色
# ax1.legend(loc='upper left', fancybox=True, framealpha=0.5, facecolor='#e0e0e0', edgecolor='black', fontsize=14)

# # 设置标题
# ax1.set_title('MI-1-Online', fontsize=19, fontweight='bold')

# # 将横坐标标签倾斜30°
# plt.xticks(rotation=30, fontsize=16)
# ax1.set_yticklabels(ax1.get_yticks(), fontsize=17, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型




# # 在第二个子图中绘制各个被试的Acc曲线
# for subject in range(num_subjects):
#     ax2.plot(methods, acc_data[subject], marker=marker_style, label=f'Subject {subject + 1}', color=subject_colors[subject], linewidth=2)

# # 为第二个子图的最后一列添加灰色背景
# ax2.add_patch(Rectangle((10 - 0.4, 53), 0.8, 42, color='gray', alpha=0.2))

# # 设置第二个子图的属性
# ax2.set_ylim(53, 95)  # 纵坐标范围设置为54-95
# # ax2.set_xlabel('Approaches', fontsize=18)
# ax2.set_yticks(np.arange(55, 96, 5))  # 设置刻度步长为5，确保是整数
# ax2.set_ylabel('Accuracy (%)', fontsize=19, labelpad=10)

# # 调整刻度字体大小
# ax2.tick_params(axis='both', which='major', labelsize=13)

# # 添加Y轴的网格线
# ax2.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)

# # 显示图例，使用弧形框框并添加较深的灰色背景，边框为深色
# ax2.legend(loc='upper left', fancybox=True, framealpha=0.5, facecolor='#e0e0e0', edgecolor='black', ncol=2, fontsize=14)

# # 将横坐标标签倾斜30°
# plt.xticks(rotation=30, fontsize=21, fontweight='bold')
# ax2.set_yticklabels(ax2.get_yticks(), fontsize=17, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型

# # 调整布局并保存图像
# plt.tight_layout()
# plt.savefig('./001_004_sda.pdf')
# plt.show()



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 设置所有字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 假设有9个被试和11种方法
num_methods = 11  # 横坐标表示的11种方法
num_subjects = 9  # 9个被试

# 方法的名称列表
methods = ['CSP+LDA', 'EEGNet', 'DAN', 'DANN', 'JAN', 'CDAN', 'MDD', 'MCC', 'SHOT', 'ISFDA', 'SDDA(Ours)']

# 9个被试的Acc数据，保留小数点后两位，增加第11种方法的数据
acc_data = np.array([
    [63.66, 66.34, 66.48, 65.29, 66.98, 66.80, 67.50, 67.09, 65.09, 60.55, 70.73],
    [56.17, 53.61, 53.92, 55.32, 54.51, 54.63, 54.85, 55.09, 56.17, 54.72, 56.02],
    [54.94, 56.77, 57.15, 55.81, 56.54, 56.89, 55.67, 55.99, 56.40, 58.08, 57.09],
    [88.42, 89.97, 90.34, 89.83, 88.33, 89.83, 92.60, 92.83, 86.24, 87.83, 93.96],
    [75.28, 73.39, 72.85, 74.83, 74.58, 75.09, 75.28, 73.31, 74.38, 72.03, 78.25],
    [75.00, 71.40, 71.74, 70.81, 70.23, 71.42, 71.34, 70.64, 72.21, 69.42, 74.65],
    [68.75, 70.00, 71.80, 67.09, 71.40, 72.91, 70.96, 72.21, 71.42, 67.65, 72.53],
    [77.89, 76.54, 77.47, 77.23, 76.81, 76.48, 77.01, 77.25, 76.95, 75.93, 79.45],
    [74.86, 70.29, 72.18, 72.04, 70.20, 72.06, 71.19, 70.35, 68.66, 67.06, 75.44]
])

# 计算每种方法的平均Acc
mean_acc = np.mean(acc_data, axis=0)

# subject_colors to match your customization
subject_colors = ['#B42B22', '#2B4871', '#996E2E', '#E995C9', '#1E90FF', '#F6944B', '#7E4909', '#009688', '#830783']
mean_color = '#FF0000'  # Red for mean accuracy

# 创建一个包含两个子图的图形（一个用于显示平均值，一个用于显示各个被试的Acc）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

# 在第一个子图中绘制平均Acc曲线
marker_style = 'D'
ax1.plot(methods, mean_acc, marker=marker_style, label='Mean Acc', linewidth=4, alpha=0.5, color=mean_color)
ax1.set_ylim(67, 74)  # 调整纵坐标范围以获得更好的显示效果
ax1.set_ylabel('Mean Accuracy (%)', fontsize=22, labelpad=10)

# 在第一个子图上为最后一列添加灰色背景
ax1.add_patch(Rectangle((10 - 0.4, 67), 0.8, 7, color='gray', alpha=0.2))

# 设置Y轴刻度为68, 71, 74
ax1.set_yticks([67, 70, 73])

# 调整刻度字体大小
ax1.tick_params(axis='both', which='major', labelsize=22)

# 只在这几个主要刻度上绘制网格线
ax1.grid(True, which='major', axis='y', color='gray', linestyle='--', linewidth=0.7)

# 设置标题
ax1.set_title('Online: BNCI2014001→BNCI2014004', fontsize=22, fontweight='bold')

# 在第二个子图中绘制各个被试的Acc曲线
for subject in range(num_subjects):
    ax2.plot(methods, acc_data[subject], marker=marker_style, label=f'Subject {subject + 1}', color=subject_colors[subject], linewidth=2)

# 为第二个子图的最后一列添加灰色背景
ax2.add_patch(Rectangle((10 - 0.4, 53), 0.8, 42, color='gray', alpha=0.2))

# 设置第二个子图的属性
ax2.set_ylim(53, 95)
ax2.set_yticks(np.arange(55, 96, 5))  # 设置刻度步长为5，确保是整数
ax2.set_ylabel('Accuracy (%)', fontsize=22, labelpad=10)

# 调整刻度字体大小
ax2.tick_params(axis='both', which='major', labelsize=22)

# 添加Y轴的网格线
ax2.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)


# 将横坐标标签倾斜30°
plt.xticks(rotation=30, fontweight='bold')
ax1.set_yticklabels(ax1.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型
ax2.set_yticklabels(ax2.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型

# 调整布局
fig.tight_layout()

# 在两个子图的右侧放置图例
ax2.legend(
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    fancybox=True,
    framealpha=0.5,
    facecolor='#e0e0e0',
    edgecolor='black',
    ncol=1,
    fontsize=22
)

# ax1.legend(
#     loc='center left',
#     bbox_to_anchor=(1.02, 0.5),
#     fancybox=True,
#     framealpha=0.5,
#     facecolor='#e0e0e0',
#     edgecolor='black',
#     fontsize=13
# )

plt.subplots_adjust(right=0.8)  # 为图例腾出空间
plt.savefig('./001_004_sda.pdf')
plt.show()
