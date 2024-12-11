import matplotlib.pyplot as plt
import numpy as np

# 教师网络准确率数据
teacher_acc = [50, 70, 80, 83, 85, 86, 87, 88, 89, 90, 91, 92, 92, 93, 93, 93, 93, 93, 93, 93]
teacher_acc = [teacher_acc[i//5] for i in range(100)]  # 每5个epoch重复一次

# 学生网络准确率数据
student_acc = [50, 60, 70, 73, 75, 76, 77, 78, 79, 80, 81, 81, 81, 81, 81, 81, 82, 82, 82, 82]
student_acc = [student_acc[i//5] for i in range(100)]


# Epoch 数组
epochs = list(range(1, 101))  # 1-20 个 epoch


# 生成随机的波动数据
np.random.seed(0)
teacher_noise = np.random.normal(loc=0, scale=0.5, size=len(epochs))
student_noise = np.random.normal(loc=0, scale=0.5, size=len(epochs))

teacher_acc_noisy = teacher_acc + teacher_noise
student_acc_noisy = student_acc + student_noise

# 计算最终的100个点的纵坐标并存储到列表中
teacher_final_points = teacher_acc_noisy[-100:]
student_final_points = student_acc_noisy[-100:]

plt.figure(figsize=(10, 6))
plt.plot(epochs, teacher_acc_noisy, color='blue', label='Teacher Network')
plt.plot(epochs, student_acc_noisy, color='orange', label='Student Network')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Knowledge Distillation Process with Noise')
plt.legend()
plt.grid(True)
plt.show()


