import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# 数据
data = [(2019.7, 13), (2020.3, 19), (2021.5, 47), (2022.2, 200), (2022.9, 510), (2023.3, 1200), (2023.9, 2322), (2024.4, 5019)]
model_names = ['T5', 'GPT-3', 'Codex', 'InstructGPT', 'ChatGPT', 'GPT-4', 'ChatGLM', 'LLama-3']
years = [2019, 2020, 2021, 2022, 2023, 2024]
values = [13, 19, 47, 200, 510, 1200, 2322, 5019]

# 插值
x = np.array([d[0] for d in data])
y = np.array([d[1] for d in data])
f = PchipInterpolator(x, y)
x_new = np.linspace(2019.7, 2024.5, 1000)
y_smooth = f(x_new)

# 限制曲线不贯穿五角星
model5_idx = model_names.index('ChatGPT')
x_start = x[model5_idx]
x_end = x[model5_idx + 1]
mask = (x_new < x_start) | (x_new > x_end)
x_new_masked = x_new[mask]
y_smooth_masked = y_smooth[mask]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x_new_masked, y_smooth_masked, color='darkorange', label='Large language model', linewidth=3)
for i, (xi, yi) in enumerate(data):
    if i == model5_idx:
        plt.scatter(xi, yi, color='blue', marker='*', s=200, zorder=5)
    else:
        plt.scatter(xi, yi, color='black', marker='o', s=60, zorder=5)
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.annotate(model_names[i], (xi, yi), textcoords="offset points", xytext=(-10, 10), ha='center', fontsize=14)
    # plt.text(xi, yi, model_names[i], ha='right', va='bottom', fontsize=14)
plt.xticks(years)
# plt.xlabel('Year')
plt.xlabel('Year', fontsize=12, fontstyle='italic')
plt.ylabel('Count', fontsize=12, fontstyle='italic')
plt.title('LLM Model Count Over Time')
plt.grid(True)
plt.legend()
plt.savefig('llm_num_visual.png')
plt.savefig('llm_num_visual.svg')
plt.show()
