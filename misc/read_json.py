import json

# 文件路径
file_path = 'results/egoschema/videotree_gpt4_qa.json'

# 读取JSON文件
with open(file_path, 'r') as file:
    data = json.load(file)

# 统计键值对数量
key_value_count = sum(len(item) for item in data.values())

print(f"Total number of key-value pairs: {key_value_count}")
# 448 => 35 RMB