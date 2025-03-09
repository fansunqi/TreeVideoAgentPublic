import pandas as pd
from collections import Counter

# 读取CSV文件
file_path = 'data/nextqa/val.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 统计type列中不同类型的数量
type_counts = Counter(data['type'])

# 输出结果
for type, count in type_counts.items():
    print(f"Type: {type}, Count: {count}")

# 如果需要，可以将结果保存到文件
with open('type_counts.txt', 'w') as f:
    for type, count in type_counts.items():
        f.write(f"Type: {type}, Count: {count}\n")
