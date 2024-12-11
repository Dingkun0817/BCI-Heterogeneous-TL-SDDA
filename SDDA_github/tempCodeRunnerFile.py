import matplotlib.pyplot as plt
import numpy as np

# 假设有9个被试和11种方法
num_methods = 11  # 横坐标表示的11种方法
num_subjects = 9  # 9个被试

# 方法的名称列表
methods = ['CSP+LDA', 'EEGNet', 'DAN', 'DANN', 'JAN', 'CDAN', 'MDD', 'MCC', 'SHOT', 'ISFDA', 'MAD(Ours)']

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

# 使用较为深一些的颜色进行润色
subject_colors = [
    '#87CEEB', '#4E8F89', '#824028', '#8EB69C', '#2F4F4F', '#FF6347', '#4682B4', '#FFD700', '#A0522D'
]
mean_color = '#FF0000'  # 深红色表示平均值

# 创建图形
plt.figure(figsize=(12, 8))

# 为每个被试绘制一条折线，使用深色配色
for subject in range(num_subjects):
    plt.plot(methods, acc_data[subject], marker='o', label=f'Subject {subject+1}', color=subject_colors[subject], linewidth=2, alpha=0.9)

# 绘制平均值折线，使用深红色
plt.plot(methods, mean_acc, marker='o', label='Mean Acc', linewidth=6, color=mean_color)

# 设置图形属性
plt.ylim(50, 100)  # 纵坐标范围设置为55-100
plt.xlabel('Methods', fontsize=14)
plt.ylabel('Accuracy (Acc)', fontsize=14)
plt.title('Acc of Different Subjects and Mean Using 11 Methods', fontsize=16)

# 设置横坐标标签的旋转角度，避免标签重叠
plt.xticks(rotation=45, ha='right')

# 显示图例
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 通过bbox_to_anchor调整图例位置

plt.tight_layout()
plt.savefig('./001_004_sda.pdf')
plt.show()
