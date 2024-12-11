import matplotlib.pyplot as plt
import numpy as np

# 消融设置名称
settings = ['Ours(Base)', 'Ours(FMA)', 'Ours(CL)', 'Ours(FMA+CL)', 'Ours(SA)', 'Ours(Full)']

# 6组实验的结果和标准差
means_list = [
    [71.00, 71.28, 73.94, 74.40, 72.19, 75.10],  #  1-UDA
    [69.04, 70.14, 73.70, 75.03, 70.47, 76.23],    # 2-UDA
    [71.46, 72.30, 75.64, 76.39, 74.02, 76.70],   # 3-UDA
    [69.77, 71.51, 72.02, 72.31, 72.88, 73.12],   # 1-SDA
    [69.24, 69.96, 70.04, 70.40, 70.46, 70.97],  # 2-SDA
    [78.02, 78.17, 78.16, 78.48, 78.72, 79.47]  # 3-SDA
]

std_devs_list = [
    [0.07, 0.37, 0.16, 0.20, 0.30, 0.31],
    [1.45, 0.90, 0.69, 0.83, 0.49, 0.50],
    [0.23, 0.39, 0.19, 0.09, 0.23, 0.12],
    [1.25, 0.48, 0.31, 0.37, 0.15, 0.34],
    [0.95, 0.63, 0.94, 0.60, 0.94, 0.52],
    [0.23, 0.34, 0.22, 0.32, 0.25, 0.37]
]

# 使用更加柔和但不透明的配色，符合Nature风格
# colors = ['#FCB2AF', '#9BDFDF', '#FFE2CE', '#C4D8E9', '#BEBCDF', '#E1D7D0']
# colors = ['#E48D8A', '#7FC7C7', '#EBC9A2', '#A2BBCE', '#9D93BD', '#C3B3A6']
colors = ['#D86F6C', '#6FA7A7', '#D8A27A', '#8AA0B5', '#8376A5', '#A79787']



# 创建2行3列的子图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 用于标注每个子图的标题
experiment_titles = ['MI-1-Offline', 'MI-2-Offline', 'P300-Offline', 'MI-1-Online', 'MI-2-Online', 'P300-Online']

# 自定义每个子图的纵坐标范围
ylims = [(70, 76), (68, 77), (70, 77), (68, 74), (67, 71.8), (77, 80)]

# 绘制每个子图
for i, ax in enumerate(axes.flat):
    # 当前子图的数据
    means = means_list[i]
    std_devs = std_devs_list[i]
    
    # 绘制柱状图，增加颜色的饱和度，通过减少alpha或移除它
    bars = ax.bar(settings, means, yerr=std_devs, capsize=5, color=colors, edgecolor='black')  # alpha设置为1，颜色更不透明
    
    # 设置横坐标文字倾斜显示
    ax.set_xticks(np.arange(len(settings)))
    ax.set_xticklabels(settings, rotation=20, ha='right', fontsize=14, fontdict={'family': 'Times New Roman'})
    ax.tick_params(axis='y', labelsize=18)  # 修改纵坐标刻度文字字体大小

    # 设置图形属性
    ax.set_ylim(ylims[i])  # 根据预定义范围设置每个子图的纵坐标
    ax.set_ylabel('Accuracy (Acc)', fontsize=18, fontdict={'family': 'Times New Roman'})
    ax.set_yticklabels(ax.get_yticks(), fontsize=14, fontdict={'family': 'Times New Roman'})  # 设置 y 轴刻度字体大小和字体类型

    # 显示标准差的小"工"字形标记
    ax.errorbar(settings, means, yerr=std_devs, fmt='none', ecolor='black', capsize=5)
    
    # 添加网格线，设置为浅灰色的虚线
    ax.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)

# 调整子图布局，防止重叠
plt.subplots_adjust(hspace=1.2)  # 增加子图之间的垂直间距，防止重叠

# 调整布局以适应所有子图
plt.tight_layout()

plt.savefig('./ablation.pdf')
plt.show()
