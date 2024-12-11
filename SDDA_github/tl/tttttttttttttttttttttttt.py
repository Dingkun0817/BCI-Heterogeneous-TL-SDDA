import numpy as np

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
    return similarity


def weighted_sum(sequences, similarities):
    # Reshape the similarities to (5, 1) to make it compatible for broadcasting
    normalized_similarities = similarities.reshape(-1, 1)

    # Use the normalized similarities for weighted sum
    weighted_sum_sequence = np.sum(sequences * normalized_similarities, axis=0)
    return weighted_sum_sequence


# 示例信号，可以替换为实际的信号数据
original_signal = np.random.rand(1, 1001)

# 步长设定为10
step_size = 10

'''
  moved_sequences是长度为100的list, 每个元素的shape为(1, 1001)
'''
moved_sequences = generate_moved_sequences(original_signal, step_size)


# 计算相似度并取前5大的序列
similarities = [calculate_similarity(original_signal, sequence) for sequence in moved_sequences]
top5_indices = np.argsort(similarities)[-5:][::-1]
top5_sequences = [moved_sequences[i] for i in top5_indices]
top5_similarities = [similarities[i] for i in top5_indices]

# 打印前5大的序列及相似度
for i, (sequence, similarity) in enumerate(zip(top5_sequences, top5_similarities), 1):
    print(f"Top {i} Sequence with Similarity {similarity}: {sequence}")

# 计算加权求和序列
final_sequence = weighted_sum(np.vstack(top5_sequences), np.array(top5_similarities))

print(final_sequence.shape)


