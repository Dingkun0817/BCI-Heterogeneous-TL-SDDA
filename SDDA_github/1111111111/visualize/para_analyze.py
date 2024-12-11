import matplotlib.pyplot as plt

# 假设这里是你的四个指标在不同 alpha 和 beta 值下的数值
# 为了简单起见，这里使用随机数生成示例数据
import numpy as np
np.random.seed(0)
alpha_values = np.linspace(0.3, 0.7, num=9)
beta_values = np.arange(4, 9)
accuracy_values = [[0.85, 0.86, 0.87, 0.88, 0.89],
                   [0.85, 0.86, 74.97306, 0.88, 0.89],
                   [0.85, 0.86, 75.04456, 0.88, 0.89],
                   [0.85, 0.86, 75.06712, 0.88, 0.89],
                   [74.98659, 0.86, 75.08332, 75.07938, 0.89],
                   [0.85, 74.98516, 75.02001, 0.88, 74.95757],
                   [0.85, 0.86, 74.86048, 0.88, 0.89],
                   [0.85, 0.86, 74.70424, 0.88, 0.89],
                   [0.85, 0.86, 74.65625, 0.88, 0.89]]
precision_values = [[0.85, 0.86, 0.87, 0.88, 0.89],
                   [0.85, 0.86, 76.97603, 0.88, 0.89],
                   [0.85, 0.86, 77.12109, 0.88, 0.89],
                   [0.85, 0.86, 77.23953, 0.88, 0.89],
                   [77.19407, 0.86, 77.34127, 77.33366, 0.89],
                   [0.85, 77.36683, 77.39944, 0.88, 77.36631],
                   [0.85, 0.86, 77.35585, 0.88, 0.89],
                   [0.85, 0.86, 77.30554, 0.88, 0.89],
                   [0.85, 0.86, 77.33454, 0.88, 0.89]]
recall_values = [[0.85, 0.86, 0.87, 0.88, 0.89],
                   [0.85, 0.86, 71.69527, 0.88, 0.89],
                   [0.85, 0.86, 71.59842, 0.88, 0.89],
                   [0.85, 0.86, 71.48301, 0.88, 0.89],
                   [71.35245, 0.86, 71.40772, 71.3874, 0.89],
                   [0.85, 71.1003, 71.11183, 0.88, 70.99956],
                   [0.85, 0.86, 70.79198, 0.88, 0.89],
                   [0.85, 0.86, 70.44509, 0.88, 0.89],
                   [0.85, 0.86, 70.23959, 0.88, 0.89]]
f1_score_values = [[0.85, 0.86, 0.87, 0.88, 0.89],
                   [0.85, 0.86, 74.08595, 0.88, 0.89],
                   [0.85, 0.86, 74.10253, 0.88, 0.89],
                   [0.85, 0.86, 74.08304, 0.88, 0.89],
                   [73.9882, 0.86, 74.09436, 74.08269, 0.89],
                   [0.85, 73.93973, 73.96347, 0.88, 73.89884],
                   [0.85, 0.86, 73.77767, 0.88, 0.89],
                   [0.85, 0.86, 73.56169, 0.88, 0.89],
                   [0.85, 0.86, 73.46004, 0.88, 0.89]]
accuracy_values = np.array(accuracy_values)
precision_values = np.array(precision_values)
recall_values = np.array(recall_values)
f1_score_values = np.array(f1_score_values)


plt.figure(figsize=(16, 12))

# 绘制 accuracy
plt.subplot(4, 2, 1)
for i in range(5):
    plt.plot(alpha_values, accuracy_values[:, i], marker='o', label=f'Beta={beta_values[i]}')
plt.title('Accuracy')
plt.xlabel('Alpha')
plt.ylabel('Value')
plt.legend()

# 绘制 precision
plt.subplot(4, 2, 2)
for i in range(5):
    plt.plot(alpha_values, precision_values[:, i], marker='o', label=f'Beta={beta_values[i]}')
plt.title('Precision')
plt.xlabel('Alpha')
plt.ylabel('Value')
plt.legend()

# 绘制 recall
plt.subplot(4, 2, 3)
for i in range(5):
    plt.plot(alpha_values, recall_values[:, i], marker='o', label=f'Beta={beta_values[i]}')
plt.title('Recall')
plt.xlabel('Alpha')
plt.ylabel('Value')
plt.legend()

# 绘制 f1-score
plt.subplot(4, 2, 4)
for i in range(5):
    plt.plot(alpha_values, f1_score_values[:, i], marker='o', label=f'Beta={beta_values[i]}')
plt.title('F1-score')
plt.xlabel('Alpha')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()
