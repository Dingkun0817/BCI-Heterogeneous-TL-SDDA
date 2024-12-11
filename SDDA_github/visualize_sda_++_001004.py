
# import matplotlib.pyplot as plt
# import numpy as np

# # 方法的名称列表
# methods = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# # 假设有9个被试和9种方法
# num_methods = 9  # 横坐标表示的9种方法
# num_subjects = 9  # 9个被试

# # 9个被试的Acc数据，保留小数点后两位
# acc_data = np.array([
#     [70.73, 73.93, 71.80, 72.64, 75.21, 76.59, 75.60, 77.59, 78.98],
#     [56.02, 55.84, 56.44, 55.58, 56.46, 55.90, 56.01, 55.52, 57.45],
#     [57.09, 55.88, 56.60, 57.26, 57.75, 57.69, 56.61, 56.59, 55.46],
#     [93.95, 94.91, 95.90, 96.34, 96.31, 97.45, 97.56, 97.81, 97.65],
#     [78.25, 83.17, 88.01, 90.42, 90.83, 91.13, 92.33, 90.91, 93.19],
#     [74.65, 77.32, 80.00, 80.44, 80.82, 81.06, 83.55, 83.49, 85.28],
#     [72.53, 75.49, 75.90, 77.57, 79.18, 80.15, 81.53, 82.28, 82.36],
#     [79.45, 81.18, 81.81, 82.82, 84.53, 85.78, 86.72, 88.33, 90.81],
#     [75.44, 78.87, 79.68, 80.51, 81.89, 82.84, 83.15, 84.61, 86.71]
# ])

# # 计算每种方法的平均Acc
# mean_acc = np.mean(acc_data, axis=0)

# # 设置被试的颜色
# subject_colors = ['#D86F6C', '#E59400', '#5F9EA0', '#4F6D7A', '#6A5ACD', '#A79787', '#B5651D', '#4682B4', '#C71585']
# mean_color = '#FF0000'  # 红色表示平均值

# # 创建一个包含两个子图的图形（一个用于显示平均值，一个用于显示各个被试的Acc）
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

# # 在第一个子图中绘制平均Acc曲线
# marker_style = 'D'
# ax1.plot(methods, mean_acc, marker=marker_style, label='Mean Acc', linewidth=5, alpha=0.5, color=mean_color)
# ax1.set_ylim(72.5, 81.5)  # 调整纵坐标范围
# ax1.set_ylabel('Mean Accuracy (%)', fontsize=16)

# # 设置Y轴刻度为72, 76, 80
# ax1.set_yticks([72, 76, 80])

# # 只在这几个主要刻度上绘制网格线
# ax1.grid(True, which='major', axis='y', color='gray', linestyle='--', linewidth=0.7)

# # 将图例移到左上角
# ax1.legend(loc='upper left')

# # 设置标题
# # ax1.set_title('Acc of Different Subjects and Mean Using 10 Methods', fontsize=16)

# # 在第二个子图中绘制各个被试的Acc曲线
# for subject in range(num_subjects):
#     ax2.plot(methods, acc_data[subject], marker=marker_style, label=f'Subject {subject + 1}', color=subject_colors[subject], linewidth=3)

# # 设置第二个子图的属性
# ax2.set_ylim(54, 100)
# ax2.set_xlabel('The number of batches of labeled target domain trials', fontsize=16)
# ax2.set_ylabel('Accuracy (%)', fontsize=16)
# ax2.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)
# ax2.legend(loc='upper left', framealpha=0.5)

# # 调整布局并保存图像
# plt.tight_layout()
# plt.savefig('./001_004_ssssssda.pdf')
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # 方法的名称列表
# methods = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# # 假设有9个被试和9种方法
# num_methods = 9  # 横坐标表示的9种方法
# num_subjects = 9  # 9个被试

# # 9个被试的Acc数据，保留小数点后两位
# acc_data = np.array([
#     [70.73, 73.93, 71.80, 72.64, 75.21, 76.59, 75.60, 77.59, 78.98],
#     [56.02, 55.84, 56.44, 55.58, 56.46, 55.90, 56.01, 55.52, 57.45],
#     [57.09, 55.88, 56.60, 57.26, 57.75, 57.69, 56.61, 56.59, 55.46],
#     [93.95, 94.91, 95.90, 96.34, 96.31, 97.45, 97.56, 97.81, 97.65],
#     [78.25, 83.17, 88.01, 90.42, 90.83, 91.13, 92.33, 90.91, 93.19],
#     [74.65, 77.32, 80.00, 80.44, 80.82, 81.06, 83.55, 83.49, 85.28],
#     [72.53, 75.49, 75.90, 77.57, 79.18, 80.15, 81.53, 82.28, 82.36],
#     [79.45, 81.18, 81.81, 82.82, 84.53, 85.78, 86.72, 88.33, 90.81],
#     [75.44, 78.87, 79.68, 80.51, 81.89, 82.84, 83.15, 84.61, 86.71]
# ])

# # 计算每种方法的平均Acc
# mean_acc = np.mean(acc_data, axis=0)

# # 设置被试的颜色
# subject_colors = ['#D86F6C', '#E59400', '#5F9EA0', '#4F6D7A', '#6A5ACD', '#A79787', '#B5651D', '#4682B4', '#C71585']
# mean_color = '#FF0000'  # 红色表示平均值


# # 设置全局字体为Times New Roman
# plt.rc('font', family='Times New Roman')
# plt.rcParams.update({'font.size': 24})  # 全局设置字体大小



# # 创建一个包含两个子图的图形（一个用于显示平均值，一个用于显示各个被试的Acc）
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

# # 在第一个子图中绘制平均Acc曲线
# marker_style = 'D'
# ax1.plot(methods, mean_acc, marker=marker_style, label='Mean Acc', linewidth=5, alpha=0.5, color=mean_color, markersize=7)
# ax1.set_ylim(72.5, 81.5)  # 调整纵坐标范围
# ax1.set_ylabel('Mean Accuracy (%)', fontsize=24, labelpad=10)

# # 设置Y轴刻度为72, 76, 80
# ax1.set_yticks([72, 76, 80])

# # 只在这几个主要刻度上绘制网格线
# ax1.grid(True, which='major', axis='y', color='gray', linestyle='--', linewidth=0.7)

# '''
# 删除图例
# '''
# # 将图例移到左上角
# # ax1.legend(loc='upper left')

# # 在第二个子图中绘制各个被试的Acc曲线
# for subject in range(num_subjects):
#     ax2.plot(methods, acc_data[subject], marker=marker_style, label=f'Subject {subject + 1}', color=subject_colors[subject], linewidth=3, markersize=7)

# # 设置第二个子图的属性
# ax2.set_ylim(54, 100)
# ax2.set_xlabel('The number of batches of labeled target domain trials', fontsize=24)
# ax2.set_ylabel('Accuracy (%)', fontsize=24, labelpad=0)
# ax2.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)

# # 将图例移到图的右侧
# ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=19, facecolor='#e0e0e0', edgecolor='black', framealpha=0.5)

# # 调整布局并保存图像
# plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例留出空间
# plt.savefig('./001_004_ssssssda.eps')
# plt.show()


import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": 'Times New Roman',
    "font.size": 24,
})

METHODS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
# 设置被试的颜色

NUM_METHODS = 9  # 横坐标表示的9种方法
NUM_SUBJECTS = 9  # 9个被试

DATA = np.array([
    [70.73, 73.93, 71.80, 72.64, 75.21, 76.59, 75.60, 77.59, 78.98],
    [56.02, 55.84, 56.44, 55.58, 56.46, 55.90, 56.01, 55.52, 57.45],
    [57.09, 55.88, 56.60, 57.26, 57.75, 57.69, 56.61, 56.59, 55.46],
    [93.95, 94.91, 95.90, 96.34, 96.31, 97.45, 97.56, 97.81, 97.65],
    [78.25, 83.17, 88.01, 90.42, 90.83, 91.13, 92.33, 90.91, 93.19],
    [74.65, 77.32, 80.00, 80.44, 80.82, 81.06, 83.55, 83.49, 85.28],
    [72.53, 75.49, 75.90, 77.57, 79.18, 80.15, 81.53, 82.28, 82.36],
    [79.45, 81.18, 81.81, 82.82, 84.53, 85.78, 86.72, 88.33, 90.81],
    [75.44, 78.87, 79.68, 80.51, 81.89, 82.84, 83.15, 84.61, 86.71]
])
SUBJECT_COLOR = ['#D86F6C', '#E59400', '#5F9EA0', '#4F6D7A', '#6A5ACD', '#A79787', '#B5651D', '#4682B4', '#C71585']

# 计算每种方法的平均Acc
MEAN = np.mean(DATA, axis=0)
MEAN_COLOR = '#FF0000'  # 红色表示平均值

if __name__ == '__main__':
    # 创建一个包含两个子图的图形（一个用于显示平均值，一个用于显示各个被试的Acc）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})

    # 在第一个子图中绘制平均Acc曲线
    marker_style = 'D'
    ax1.plot(METHODS, MEAN, marker=marker_style, label='Mean Acc', linewidth=5, alpha=0.5, color=MEAN_COLOR)
    ax1.set_ylim(72.5, 81.5)  # 调整纵坐标范围
    ax1.set_ylabel('Mean Accuracy (%)', labelpad=10)

    # 设置Y轴刻度为72, 76, 80
    ax1.set_yticks([72, 76, 80])

    # 只在这几个主要刻度上绘制网格线
    ax1.grid(True, which='major', axis='y', color='gray', linestyle='--', linewidth=0.7)

    # 在第二个子图中绘制各个被试的Acc曲线
    for subject in range(NUM_SUBJECTS):
        ax2.plot(
            METHODS, DATA[subject],
            marker=marker_style, label=f'Subject {subject + 1}',
            color=SUBJECT_COLOR[subject], linewidth=3
        )

    # 设置第二个子图的属性
    ax2.set_ylim(54, 99.9)

    ax2.set_xlabel('The number of batches of labeled target domain trials', fontweight='bold', labelpad=10)
    ax2.set_ylabel('Accuracy (%)', labelpad=10)
    ax2.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)

    # 将图例移到图的右侧
    # ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.035), ncol=2, facecolor='#e0e0e0', edgecolor='black', framealpha=0.5)
    ax2.legend(
        loc='lower left', ncol=3,
        bbox_to_anchor=(0.0, 0.1), facecolor='#e0e0e0', edgecolor='black', framealpha=0.5, fontsize=16
    )

    # 调整布局并保存图像
    plt.tight_layout(rect=[0, 0, 1, 1])  # 为图例留出空间
    plt.savefig('./001_004_ssssssda.pdf')
    plt.show()
