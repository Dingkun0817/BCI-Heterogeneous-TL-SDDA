# import matplotlib.pyplot as plt
# import numpy as np

# # 方法的名称列表
# methods = ['EEGNet', 'DAN', 'DANN', 'JAN', 'CDAN', 'MDD', 'MCC', 'SHOT', 'ISFDA', 'MAD(Ours)']

# # 假设有14个被试和10种方法
# num_methods = 10  # 横坐标表示的10种方法
# num_subjects = 14  # 14个被试

# # 14个被试的Acc数据，保留小数点后两位
# acc_data = np.array([
#     [68.40, 69.20, 68.40, 72.80, 69.00, 71.40, 71.40, 68.60, 67.80, 74.00],
#     [76.00, 77.40, 70.20, 76.60, 68.20, 78.20, 78.20, 81.00, 76.80, 77.00],
#     [73.20, 68.40, 63.80, 76.60, 86.40, 67.20, 96.60, 66.20, 64.60, 98.40],
#     [71.80, 66.80, 72.20, 70.20, 70.60, 73.60, 69.80, 69.80, 71.60, 75.40],
#     [73.80, 76.20, 77.60, 78.40, 80.00, 74.60, 83.00, 79.60, 73.80, 86.60],
#     [59.80, 60.00, 57.00, 61.40, 55.60, 59.00, 62.00, 59.40, 59.40, 69.60],
#     [85.00, 85.80, 83.60, 86.40, 82.20, 88.40, 89.80, 87.00, 84.60, 86.80],
#     [66.20, 67.20, 68.40, 65.20, 63.80, 64.20, 62.20, 68.20, 64.40, 79.00],
#     [86.20, 85.00, 84.40, 82.20, 82.80, 85.80, 91.00, 89.80, 83.60, 92.80],
#     [61.80, 59.80, 63.80, 65.80, 61.20, 63.60, 62.80, 62.80, 60.40, 66.80],
#     [73.60, 79.00, 76.20, 76.60, 74.80, 73.80, 80.60, 75.60, 74.00, 89.40],
#     [57.20, 58.80, 57.80, 61.60, 62.00, 61.00, 61.80, 58.60, 57.40, 63.80],
#     [56.00, 56.60, 57.00, 56.80, 55.80, 57.20, 55.40, 60.80, 59.00, 61.40],
#     [47.20, 47.20, 49.20, 50.40, 53.80, 51.80, 49.60, 51.00, 52.00, 46.20]
# ])

# # 计算每种方法的平均Acc
# mean_acc = np.mean(acc_data, axis=0)
# print(mean_acc)

# # 根据图中的SCI配色设置颜色，并为14个被试分配颜色
# subject_colors = [
#     '#4E8F89', '#EEDF70', '#E9D26F', '#D97943', '#824028', '#FF4500', '#FF8C00', '#FFD700', 
#     '#87CEEB', '#8B4513', '#4682B4', '#9932CC', '#556B2F', '#2F4F4F'
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
# plt.ylim(55, 100)  # 纵坐标范围设置为55-100
# plt.xlabel('Methods', fontsize=14)
# plt.ylabel('Accuracy (Acc)', fontsize=14)
# plt.title('Acc of Different Subjects and Mean Using 10 Methods', fontsize=16)

# # 显示图例
# plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 通过bbox_to_anchor调整图例位置

# plt.tight_layout()
# plt.savefig('./001_002_uda.pdf')
# plt.show()



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 设置所有字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 方法的名称列表
methods = ['EEGNet', 'DAN', 'DANN', 'JAN', 'CDAN', 'MDD', 'MCC', 'SHOT', 'ISFDA', 'SDDA(Ours)']

# 假设有14个被试和10种方法
num_methods = 10  # 横坐标表示的10种方法
num_subjects = 14  # 14个被试

# 14个被试的Acc数据，保留小数点后两位
acc_data = np.array([
    [68.40, 69.20, 68.40, 72.80, 69.00, 71.40, 71.40, 68.60, 67.80, 74.00],
    [76.00, 77.40, 70.20, 76.60, 68.20, 78.20, 78.20, 81.00, 76.80, 77.00],
    [73.20, 68.40, 63.80, 76.60, 86.40, 67.20, 96.60, 66.20, 64.60, 98.40],
    [71.80, 66.80, 72.20, 70.20, 70.60, 73.60, 69.80, 69.80, 71.60, 75.40],
    [73.80, 76.20, 77.60, 78.40, 80.00, 74.60, 83.00, 79.60, 73.80, 86.60],
    [59.80, 60.00, 57.00, 61.40, 55.60, 59.00, 62.00, 59.40, 59.40, 69.60],
    [85.00, 85.80, 83.60, 86.40, 82.20, 88.40, 89.80, 87.00, 84.60, 86.80],
    [66.20, 67.20, 68.40, 65.20, 63.80, 64.20, 62.20, 68.20, 64.40, 79.00],
    [86.20, 85.00, 84.40, 82.20, 82.80, 85.80, 91.00, 89.80, 83.60, 92.80],
    [61.80, 59.80, 63.80, 65.80, 61.20, 63.60, 62.80, 62.80, 60.40, 66.80],
    [73.60, 79.00, 76.20, 76.60, 74.80, 73.80, 80.60, 75.60, 74.00, 89.40],
    [57.20, 58.80, 57.80, 61.60, 62.00, 61.00, 61.80, 58.60, 57.40, 63.80],
    [56.00, 56.60, 57.00, 56.80, 55.80, 57.20, 55.40, 60.80, 59.00, 61.40],
    [47.20, 47.20, 49.20, 50.40, 53.80, 51.80, 49.60, 51.00, 52.00, 46.20]
])

# 计算每种方法的平均Acc
mean_acc = np.mean(acc_data, axis=0)
print(mean_acc)

# subject_colors to match your customization
# subject_colors = ['#B42B22', '#2B4871', '#996E2E', '#E995C9', '#1E90FF', '#F6944B', '#7E4909', '#009688', '#830783']
subject_colors = [
    '#B42B22',  # 深红色
    '#2B4871',  # 深蓝色
   '#32CD32',  # 亮绿色（新添加）
    '#E995C9',  # 粉色
    '#1E90FF',  # 天蓝色
    '#F6944B',  # 橙色
    '#7E4909',  # 深棕色
    '#009688',  # 青色
    '#830783',  # 紫色
    '#FFD700',  # 金黄色（新添加）
    '#4682B4',  # 钢蓝色（新添加）
    '#996E2E',  # 棕色
    '#FF6347',  # 番茄红色（新添加）
    '#8B4513',  # 棕褐色（新添加）
]

mean_color = '#FF0000'  # Red for mean accuracy

# 创建一个包含两个子图的图形（一个用于显示平均值，一个用于显示各个被试的Acc）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

# 在第一个子图中绘制平均Acc曲线
marker_style = 'D'
ax1.plot(methods, mean_acc, marker=marker_style, label='Mean Acc', linewidth=4, alpha=0.5, color=mean_color)
ax1.set_ylim(67, 77)  # 调整纵坐标范围以获得更好的显示效果
ax1.set_ylabel('Mean Accuracy (%)', fontsize=22, labelpad=10)

# 在第一个子图上为最后一列添加灰色背景
ax1.add_patch(Rectangle((9 - 0.4, 67), 0.8, 10, color='gray', alpha=0.2))

# 设置Y轴刻度为68, 71, 74
ax1.set_yticks([68, 71, 74])

# 调整刻度字体大小
ax1.tick_params(axis='both', which='major', labelsize=22)

# 只在这几个主要刻度上绘制网格线
ax1.grid(True, which='major', axis='y', color='gray', linestyle='--', linewidth=0.7)

# 将图例移到左上角，使用弧形框框并添加较深的灰色背景，边框为深色
# ax1.legend(loc='upper left', fancybox=True, framealpha=0.5, facecolor='#e0e0e0', edgecolor='black', fontsize=14)

# 设置标题
ax1.set_title('Offline: BNCI2015001→BNCI2014002', fontsize=22, fontweight='bold')

# 将横坐标标签倾斜30°
plt.xticks(rotation=30, fontsize=22)
ax1.set_yticklabels(ax1.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型




# 在第二个子图中绘制各个被试的Acc曲线
for subject in range(num_subjects):
    ax2.plot(methods, acc_data[subject], marker=marker_style, label=f'Subject {subject + 1}', color=subject_colors[subject], linewidth=2)

# 为第二个子图的最后一列添加灰色背景
ax2.add_patch(Rectangle((9 - 0.4, 45), 0.8, 54, color='gray', alpha=0.2))

# 设置第二个子图的属性
ax2.set_ylim(45, 99)  # 纵坐标范围设置为54-95
# ax2.set_xlabel('Approaches', fontsize=18)
ax2.set_yticks(np.arange(45, 100, 5))  # 设置刻度步长为5，确保是整数
ax2.set_ylabel('Accuracy (%)', fontsize=22, labelpad=6)

# 调整刻度字体大小
ax2.tick_params(axis='both', which='major', labelsize=22)

# 添加Y轴的网格线
ax2.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)

# 显示图例，使用弧形框框并添加较深的灰色背景，边框为深色
# ax2.legend(loc='upper left', fancybox=True, framealpha=0.5, facecolor='#e0e0e0', edgecolor='black', ncol=3, fontsize=14)

# 将横坐标标签倾斜30°
plt.xticks(rotation=30, fontsize=22, fontweight='bold')
ax1.set_yticklabels(ax1.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 x 轴刻度字体大小和字体类型
ax2.set_yticklabels(ax2.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型

# 调整布局并保存图像
plt.tight_layout()
plt.savefig('./001_002_uda.pdf')
plt.show()
