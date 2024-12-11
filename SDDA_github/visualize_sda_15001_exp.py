# import matplotlib.pyplot as plt
# import numpy as np

# # 方法的名称列表
# methods = ['CSP+LDA', 'EEGNet', 'DAN', 'DANN', 'JAN', 'CDAN', 'MDD', 'MCC', 'SHOT', 'ISFDA', 'MAD(Ours)']

# # 假设有14个被试和10种方法
# num_methods = 11  # 横坐标表示的10种方法
# num_subjects = 14  # 14个被试

# # 14个被试的Acc数据，保留小数点后两位
# acc_data = np.array([
#     [58.82, 69.71, 67.06, 73.82, 70.88, 65.29, 70.29, 67.65, 71.18, 70.88, 67.94],
#     [72.06, 75.29, 74.41, 73.82, 79.12, 71.76, 77.06, 80.59, 78.82, 76.76, 73.24],
#     [91.18, 91.47, 90.29, 95.00, 80.59, 95.59, 90.88, 91.47, 60.59, 62.65, 94.12],
#     [64.71, 66.47, 65.29, 69.71, 73.53, 74.12, 72.06, 72.65, 70.29, 74.71, 72.94],
#     [77.94, 71.47, 73.82, 70.59, 71.18, 65.59, 71.47, 73.24, 73.24, 73.24, 76.47],
#     [60.29, 61.47, 58.24, 58.24, 50.59, 55.00, 55.00, 53.53, 56.18, 57.35, 58.82],
#     [85.29, 81.47, 81.76, 81.47, 83.82, 78.24, 85.29, 79.12, 83.53, 79.71, 84.12],
#     [77.94, 63.53, 63.53, 62.35, 64.41, 58.82, 68.82, 64.12, 64.12, 63.82, 68.24],
#     [92.65, 84.41, 87.06, 90.88, 85.88, 85.00, 86.47, 87.65, 86.18, 81.18, 87.65],
#     [55.88, 62.06, 60.29, 59.71, 63.53, 57.94, 57.94, 61.76, 60.29, 60.00, 61.18],
#     [60.29, 70.59, 72.35, 68.53, 72.94, 64.71, 72.65, 70.88, 67.65, 67.65, 82.06],
#     [60.29, 53.24, 57.35, 53.24, 54.12, 56.47, 52.06, 54.41, 59.12, 55.88, 57.65],
#     [45.59, 52.35, 54.71, 50.88, 52.06, 54.12, 47.06, 52.94, 56.18, 57.06, 53.53],
#     [42.65, 47.94, 51.76, 50.00, 51.18, 55.88, 49.71, 48.82, 57.94, 57.65, 55.59]
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
# plt.ylim(45, 100)  # 纵坐标范围设置为55-100
# plt.xlabel('Methods', fontsize=14)
# plt.ylabel('Accuracy (Acc)', fontsize=14)
# plt.title('Acc of Different Subjects and Mean Using 11 Methods', fontsize=16)

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
methods = ['CSP+LDA', 'EEGNet', 'DAN', 'DANN', 'JAN', 'CDAN', 'MDD', 'MCC', 'SHOT', 'ISFDA', 'SDDA(Ours)']

# 假设有14个被试和10种方法
num_methods = 11  # 横坐标表示的10种方法
num_subjects = 14  # 14个被试

# 14个被试的Acc数据，保留小数点后两位
acc_data = np.array([
    [58.82, 69.71, 67.06, 73.82, 70.88, 65.29, 70.29, 67.65, 71.18, 70.88, 67.94],
    [72.06, 75.29, 74.41, 73.82, 79.12, 71.76, 77.06, 80.59, 78.82, 76.76, 73.24],
    [91.18, 91.47, 90.29, 95.00, 80.59, 95.59, 90.88, 91.47, 60.59, 62.65, 94.12],
    [64.71, 66.47, 65.29, 69.71, 73.53, 74.12, 72.06, 72.65, 70.29, 74.71, 72.94],
    [77.94, 71.47, 73.82, 70.59, 71.18, 65.59, 71.47, 73.24, 73.24, 73.24, 76.47],
    [60.29, 61.47, 58.24, 58.24, 50.59, 55.00, 55.00, 53.53, 56.18, 57.35, 58.82],
    [85.29, 81.47, 81.76, 81.47, 83.82, 78.24, 85.29, 79.12, 83.53, 79.71, 84.12],
    [77.94, 63.53, 63.53, 62.35, 64.41, 58.82, 68.82, 64.12, 64.12, 63.82, 68.24],
    [92.65, 84.41, 87.06, 90.88, 85.88, 85.00, 86.47, 87.65, 86.18, 81.18, 87.65],
    [55.88, 62.06, 60.29, 59.71, 63.53, 57.94, 57.94, 61.76, 60.29, 60.00, 61.18],
    [60.29, 70.59, 72.35, 68.53, 72.94, 64.71, 72.65, 70.88, 67.65, 67.65, 82.06],
    [60.29, 53.24, 57.35, 53.24, 54.12, 56.47, 52.06, 54.41, 59.12, 55.88, 57.65],
    [45.59, 52.35, 54.71, 50.88, 52.06, 54.12, 47.06, 52.94, 56.18, 57.06, 53.53],
    [42.65, 47.94, 51.76, 50.00, 51.18, 55.88, 49.71, 48.82, 57.94, 57.65, 55.59]
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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

# 在第一个子图中绘制平均Acc曲线
marker_style = 'D'
ax1.plot(methods, mean_acc, marker=marker_style, label='Mean Acc', linewidth=4, alpha=0.5, color=mean_color)
ax1.set_ylim(66.5, 71.5)  # 调整纵坐标范围以获得更好的显示效果
ax1.set_ylabel('Mean Accuracy (%)', fontsize=22, labelpad=10)

# 在第一个子图上为最后一列添加灰色背景
ax1.add_patch(Rectangle((10 - 0.4, 66.5), 0.8, 5, color='gray', alpha=0.2))

# 设置Y轴刻度为68, 71, 74
ax1.set_yticks([67, 69, 71])

# 调整刻度字体大小
ax1.tick_params(axis='both', which='major', labelsize=22)

# 只在这几个主要刻度上绘制网格线
ax1.grid(True, which='major', axis='y', color='gray', linestyle='--', linewidth=0.7)

# 将图例移到左上角，使用弧形框框并添加较深的灰色背景，边框为深色
# ax1.legend(loc='upper left', fancybox=True, framealpha=0.5, facecolor='#e0e0e0', edgecolor='black', fontsize=22)

# 设置标题
ax1.set_title('Online: BNCI2015001→BNCI2014002', fontsize=22, fontweight='bold')

# 将横坐标标签倾斜30°
plt.xticks(rotation=30, fontsize=20)
ax1.set_yticklabels(ax1.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型




# 在第二个子图中绘制各个被试的Acc曲线
for subject in range(num_subjects):
    ax2.plot(methods, acc_data[subject], marker=marker_style, label=f'Subject {subject + 1}', color=subject_colors[subject], linewidth=2)

# 为第二个子图的最后一列添加灰色背景
ax2.add_patch(Rectangle((10 - 0.4, 42), 0.8, 58, color='gray', alpha=0.2))

# 设置第二个子图的属性
ax2.set_ylim(42, 100)  # 纵坐标范围设置为54-95
# ax2.set_xlabel('Approaches', fontsize=18)
ax2.set_yticks(np.arange(45, 100, 5))  # 设置刻度步长为5，确保是整数
ax2.set_ylabel('Accuracy (%)', fontsize=22, labelpad=10)

# 调整刻度字体大小
ax2.tick_params(axis='both', which='major', labelsize=13)

# 添加Y轴的网格线
ax2.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)

# 显示图例，使用弧形框框并添加较深的灰色背景，边框为深色
# ax2.legend(loc='upper left', fancybox=True, framealpha=0.5, facecolor='#e0e0e0', edgecolor='black', ncol=3, fontsize=14)

# 将横坐标标签倾斜30°
plt.xticks(rotation=30, fontsize=21, fontweight='bold')
ax1.set_yticklabels(ax1.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型
ax2.set_yticklabels(ax2.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman', 'weight': 'bold'})  # 设置 y 轴刻度字体大小和字体类型


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


plt.subplots_adjust(right=0.8)  # 为图例腾出空间


# 调整布局并保存图像
plt.tight_layout()
plt.savefig('./001_002_sda.pdf')
plt.show()
