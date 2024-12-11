import matplotlib.pyplot as plt
import numpy as np

# 消融设置名称
settings = ['Ours(Base)', 'Ours(SD)', 'Ours(MA)', 'Ours(CL)', 'Ours(MA+CL)', 'Ours(Full)']

# 6组实验的结果和标准差
means_list = [
    [71.00, 72.19, 71.28, 73.94, 74.40, 75.10],  #  1-UDA
    [69.04, 70.47, 70.14, 73.70, 75.03, 76.23],    # 2-UDA
    [71.46, 74.02, 72.30, 75.64, 76.39, 76.70],   # 3-UDA
    [69.77, 72.88, 71.51, 72.02, 72.31, 73.12],   # 1-SDA
    [69.24, 70.46, 69.96, 70.04, 70.40, 70.97],  # 2-SDA
    [78.02, 78.72, 78.17, 78.16, 78.48, 79.47]  # 3-SDA
]

std_devs_list = [
    [0.07, 0.30, 0.37, 0.16, 0.20, 0.31],
    [1.45, 0.49, 0.90, 0.69, 0.83, 0.50],
    [0.23, 0.23, 0.39, 0.19, 0.09, 0.12],
    [1.25, 0.15, 0.48, 0.31, 0.37, 0.34],
    [0.95, 0.94, 0.63, 0.94, 0.60, 0.52],
    [0.23, 0.25, 0.34, 0.22, 0.32, 0.37]
]

# 颜色设置
# colors = ['#D86F6C', '#6FA7A7', '#D8A27A', '#8AA0B5', '#8376A5', '#A79787']
colors = ['#BEE8E8', '#9FD1EE', '#87CEEB', '#4C9BE6', '#8C9FCA', '#4D4D9F']

# 设置全局字体为Times New Roman
plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 24})  # 全局设置字体大小



# 创建2行3列的子图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 用于标注每个子图的标题
experiment_titles = ['Offline: 14001→14004', 'Offline: 15001→14002', 
                     'Offline: 14009→14008', 'Online: 14001→14004', 
                     'Online: 15001→14002', 'Online: 14009→14008']

# 自定义每个子图的纵坐标范围
ylims = [(70, 76), (68, 77), (70, 77), (68, 74), (67, 71.8), (77, 80)]

# 绘制每个子图
for i, ax in enumerate(axes.flat):
    # 当前子图的数据
    means = means_list[i]
    std_devs = std_devs_list[i]
    
    # 绘制柱状图，增加颜色的饱和度
    bars = ax.bar(settings, means, yerr=std_devs, capsize=5, color=colors, edgecolor='black')

    # 设置横坐标文字倾斜显示
    ax.set_xticks(np.arange(len(settings)))
    # ax.set_xticklabels(settings, rotation=20, ha='right', fontsize=14, fontdict={'family': 'Times New Roman'})
    ax.set_xticklabels(settings, rotation=26, ha='right', fontsize=14, fontdict={'family': 'Times New Roman', 'weight': 'bold'})


    # 设置图形属性
    ax.set_ylim(ylims[i])  # 根据预定义范围设置每个子图的纵坐标
    if i % 3 != 2:  # 如果不是最后一列
        ax.set_ylabel('Accuracy (%)', fontsize=22, fontdict={'family': 'Times New Roman'})
    else:  # 最后一列使用AUC
        ax.set_ylabel('AUC (%)', fontsize=22, fontdict={'family': 'Times New Roman'})
    # ax.set_ylabel('Accuracy (%)', fontsize=22, fontdict={'family': 'Times New Roman'})
    ax.set_title(experiment_titles[i], fontsize=24, fontdict={'family': 'Times New Roman'})  # 使用标题

    # 设置刻度字体大小和字体类型
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_yticklabels(ax.get_yticks(), fontsize=22, fontdict={'family': 'Times New Roman'})  # 设置 y 轴刻度字体大小和字体类型

    # 显示标准差的小"工"字形标记
    ax.errorbar(settings, means, yerr=std_devs, fmt='none', ecolor='black', capsize=5)
    
    # 添加网格线，设置为浅灰色的虚线
    ax.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)

# 调整子图布局，防止重叠
plt.subplots_adjust(hspace=1.8, wspace=1.8)

# 调整布局以适应所有子图
plt.tight_layout()

plt.savefig('./ablation.pdf')
plt.show()
