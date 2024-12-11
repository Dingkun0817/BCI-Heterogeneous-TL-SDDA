import matplotlib.pyplot as plt
import numpy as np

# 消融设置名称
settings = ['Ours(Base)', 'Ours(FMA)', 'Ours(CL)', 'Ours(FMA+CL)', 'Ours(SA)', 'Ours(Full)']

# 实验结果和标准差
means = [69.15, 69.62, 70.54, 71.16, 70.28, 73.41]
std_devs = [0.70, 0.51, 0.57, 0.39, 0.21, 0.26]

# 使用更加柔和的配色，符合Nature风格
# colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']
# colors = ['#C8D5B9', '#B9D5BA', '#8FC0A9', '#467F79', '#F8DDDA', '#E1D7D0']
colors = ['#FCB2AF', '#9BDFDF', '#FFE2CE', '#C4D8E9', '#BEBCDF', '#E1D7D0']

# 绘制柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(settings, means, yerr=std_devs, capsize=5, color=colors, alpha=0.9, edgecolor='black')

# 设置横坐标文字倾斜显示
plt.xticks(rotation=20, ha='right', fontsize=12)

# 设置图形属性
plt.ylim(68, 74)  # 根据数据范围设置纵坐标
plt.xlabel('Ablation Settings', fontsize=14)
plt.ylabel('Accuracy (Acc)', fontsize=14)
plt.title('Ablation Study Results with Standard Deviation', fontsize=16)

# 显示标准差的小"工"字形标记
plt.errorbar(settings, means, yerr=std_devs, fmt='none', ecolor='black', capsize=5)

# 添加网格线，设置为浅灰色的虚线
plt.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)

# 显示图形
plt.tight_layout()
plt.show()
