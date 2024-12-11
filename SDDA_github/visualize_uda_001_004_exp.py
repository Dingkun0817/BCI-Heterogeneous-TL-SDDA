import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 设置所有字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 方法的名称列表
methods = ['EEGNet', 'DAN', 'DANN', 'JAN', 'CDAN', 'MDD', 'MCC', 'SHOT', 'ISFDA', 'SDDA(Ours)']

# 假设有9个被试和10种方法
num_methods = 10  # 横坐标表示的10种方法
num_subjects = 9  # 9个被试

# 9个被试的Acc数据，保留小数点后两位
acc_data = np.array([
    [66.53, 65.67, 65.08, 66.39, 65.19, 65.28, 63.44, 63.58, 64.75, 69.94],
    [55.62, 55.85, 55.21, 55.77, 56.62, 55.50, 55.18, 55.24, 56.06, 57.79],
    [57.67, 57.17, 58.03, 57.39, 58.36, 58.58, 54.47, 56.83, 58.50, 57.06],
    [84.92, 86.27, 84.78, 83.22, 85.84, 87.00, 91.95, 91.89, 84.95, 93.95],
    [74.54, 74.73, 74.16, 75.46, 75.16, 72.51, 77.95, 77.35, 71.97, 86.27],
    [68.70, 69.97, 70.25, 72.11, 73.28, 71.17, 74.33, 71.50, 67.61, 79.58],
    [67.95, 70.44, 68.44, 67.47, 69.53, 69.22, 73.47, 71.53, 68.47, 76.47],
    [75.84, 75.92, 76.71, 75.00, 75.08, 76.37, 76.16, 75.11, 75.53, 76.84],
    [70.58, 70.56, 72.53, 70.36, 71.11, 70.83, 67.92, 73.72, 70.94, 77.94]
])

# 计算每种方法的平均Acc
mean_acc = np.mean(acc_data, axis=0)

# subject_colors to match your customization
subject_colors = ['#B42B22', '#2B4871', '#996E2E', '#E995C9', '#1E90FF', '#F6944B', '#7E4909', '#009688', '#830783']
mean_color = '#FF0000'  # Red for mean accuracy

# 创建一个包含两个子图的图形（一个用于显示平均值，一个用于显示各个被试的Acc）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

# 在第一个子图中绘制平均Acc曲线
marker_style = 'D'
ax1.plot(methods, mean_acc, marker=marker_style, label='Mean Acc', linewidth=4, alpha=0.5, color=mean_color)
ax1.set_ylim(68, 76)  # 调整纵坐标范围以获得更好的显示效果
ax1.set_ylabel('Mean Accuracy (%)', fontsize=22, labelpad=10)

# 在第一个子图上为最后一列添加灰色背景
ax1.add_patch(Rectangle((9 - 0.4, 68), 0.8, 8, color='gray', alpha=0.2))

# 设置Y轴刻度为68, 71, 74
ax1.set_yticks([68, 71, 74])

# 调整刻度字体大小
ax1.tick_params(axis='both', which='major', labelsize=22)

# 只在这几个主要刻度上绘制网格线
ax1.grid(True, which='major', axis='y', color='gray', linestyle='--', linewidth=0.7)


'''
删掉不加图例
'''
# # 将图例移到左上角，使用弧形框框并添加较深的灰色背景，边框为深色
# ax1.legend(loc='upper left', fancybox=True, framealpha=0.5, facecolor='#e0e0e0', edgecolor='black', fontsize=14)

# 设置标题
ax1.set_title('Offline: BNCI2014001→BNCI2014004', fontsize=22, fontweight='bold')

# 将横坐标标签倾斜30°
plt.xticks(rotation=30, fontsize=22)
ax1.set_yticklabels(ax1.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型



# 在第二个子图中绘制各个被试的Acc曲线
for subject in range(num_subjects):
    ax2.plot(methods, acc_data[subject], marker=marker_style, label=f'Subject {subject + 1}', color=subject_colors[subject], linewidth=2)

# 为第二个子图的最后一列添加灰色背景
ax2.add_patch(Rectangle((9 - 0.4, 54), 0.8, 41, color='gray', alpha=0.2))

# 设置第二个子图的属性
ax2.set_ylim(54, 95)  # 纵坐标范围设置为54-95
# ax2.set_xlabel('Approaches', fontsize=18)
ax2.set_yticks(np.arange(55, 96, 5))  # 设置刻度步长为5，确保是整数
ax2.set_ylabel('Accuracy (%)', fontsize=22, labelpad=10)


# 调整刻度字体大小
ax2.tick_params(axis='both', which='major', labelsize=22)

# 添加Y轴的网格线
ax2.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)


'''
删掉图例
'''
# 显示图例，使用弧形框框并添加较深的灰色背景，边框为深色
# ax2.legend(loc='upper left', fancybox=True, framealpha=0.5, facecolor='#e0e0e0', edgecolor='black', ncol=2, fontsize=14)

# 将横坐标标签倾斜30°
plt.xticks(rotation=30, fontweight='bold')
ax2.set_yticklabels(ax2.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型

# 调整布局并保存图像
plt.tight_layout()
plt.savefig('./001_004_uda.pdf')
plt.show()



