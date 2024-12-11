import sys
sys.path.append('E:\Pycharm_BCI\Self_Study\T-TIME')
# sys.path.append('/data1/ldk/T-TIME')
import random
import numpy as np
import matplotlib.pyplot as plt

from tl.utils.dataloader import data_process


dataset1 = 'BNCI2014001'
X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(dataset1)


print('X_1, y_1:', X_1.shape, y_1.shape)


eeg_sample = X_1[320]

# 放大信号，通过乘以一个缩放因子
# scaling_factor = 15  # 这个值可以根据需要调整
amplified_sample = eeg_sample[0][:100] * 11  # 放大前100个点
remaining_sample = eeg_sample[0][100:300] * 7 # 保持100-500部分不变
remaining_sample2 = eeg_sample[0][300:500] * 10 # 保持100-500部分不变

# 合并放大的部分和未变化的部分
combined_sample = list(amplified_sample) + list(remaining_sample) + list(remaining_sample2)


# 绘制放大后的EEG波形
plt.figure(figsize=(13, 10))

line_color = '#1f77b4'
line_width = 2.5

plt.subplot(22, 1, 1)
plt.plot(combined_sample, color=line_color, linewidth=line_width)
plt.ylim([min(combined_sample) - 0.5, max(combined_sample) + 0.5])  # 调整y轴范围，增强可视效果
plt.axis('off')  # 隐藏坐标轴

plt.tight_layout()
plt.savefig('./333t.pdf')
plt.show()
# plt.savefig('./111.pdf')