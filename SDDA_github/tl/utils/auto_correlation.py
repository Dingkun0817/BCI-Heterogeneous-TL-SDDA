import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_moved_sequences(signal, step):
    length = signal.shape[1]
    num_sequences = length // step

    moved_sequences = []

    for i in range(num_sequences):
        start_idx = i * step
        moved_sequence = np.roll(signal, -start_idx, axis=1)
        moved_sequences.append(moved_sequence)

    return moved_sequences

def calculate_similarity(original, sequence):
    # 使用欧几里得距离计算相似度，可以根据实际情况选择其他距离度量
    distance = np.linalg.norm(original - sequence)
    similarity = 1 / (1 + distance)

    # original = torch.tensor(original)
    # sequence = torch.tensor(sequence)
    # #
    # q_fft = torch.fft.rfft(original)
    # k_fft = torch.fft.rfft(sequence)
    # res = q_fft * torch.conj(k_fft)
    # similarity = torch.fft.irfft(res, dim=-1)
    # similarity = torch.abs(similarity)
    # similarity = similarity.mean()
    # print(similarity.shape)

    return similarity

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 减去最大值，防止数值溢出
    return exp_x / np.sum(exp_x, axis=0)

def weighted_sum(sequences, similarities):
    # Reshape the similarities to (5, 1) to make it compatible for broadcasting
    normalized_similarities = similarities.reshape(-1, 1)

    # Use the normalized similarities for weighted sum
    softmax_weights = softmax(normalized_similarities.flatten())
    softmax_weights = softmax_weights.reshape(-1, 1)
    weighted_sum_sequence = np.sum(sequences * softmax_weights, axis=0)
    # print("++_++_++_+_+_+_++_++_++_+_++++")
    # print(softmax_weights)
    # print("++_++_++_+_+_+_++_++_++_+_++++")
    return weighted_sum_sequence

    # weighted_sum_sequence = np.sum(sequences * normalized_similarities, axis=0)
    # return weighted_sum_sequence

def operation_A(X):
    X = X.reshape(1, 1001)
    # print("+_+__+_+_+++_+_+_+_+__+_++_+_+_+")
    # print(X.shape)
    step_size = 100
    moved_sequences = generate_moved_sequences(X, step_size)
    # draw(X, moved_sequences)    # 可视化
    # print(len(moved_sequences))
    # print(moved_sequences[0].shape)
    similarities = [calculate_similarity(X, sequence) for sequence in moved_sequences]
    top5_indices = np.argsort(similarities)[-5:][::-1]
    top5_sequences = [moved_sequences[i] for i in top5_indices]
    top5_similarities = [similarities[i] for i in top5_indices]
    # 计算加权求和序列
    final_sequence = weighted_sum(np.vstack(top5_sequences), np.array(top5_similarities))
    X = X.reshape(1001, )
    final_sequence = 1.0*final_sequence + 0.0*X
    # print("+_+__+_+_+++_+_+_+_+__+_++_+_+_+")
    # print(final_sequence.shape)
    return final_sequence

def draw(X, XX):
    X = X.reshape(1, 1001)
    XX = np.array(XX)
    XX = XX.reshape(10, 1001)
    data = np.vstack((X, XX))

    # 创建时间轴
    time_axis = np.arange(0, 1001)

    # 创建22个子图，每个子图对应一个电极
    fig, axs = plt.subplots(11, 1, figsize=(10, 20), sharex=True)

    # 循环绘制每个电极的波形
    for electrode_index in range(11):
        electrode_data = data[electrode_index, :]
        axs[electrode_index].plot(time_axis, electrode_data)
        axs[electrode_index].set_title(f'Electrode {electrode_index + 1}')
        axs[electrode_index].set_ylabel('Amplitude')

    # 设置x轴标签
    axs[-1].set_xlabel('Time (samples)')

    # 调整子图布局，防止重叠
    plt.tight_layout()
    plt.show()