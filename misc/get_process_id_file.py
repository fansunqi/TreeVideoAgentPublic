import json

# 读取 JSON 文件
with open('/Users/sunqifan/Documents/codes/video_agents/EgoSchema/downloaded_videos.json', 'r') as file:
    data = json.load(file)

# 提取 q_uid 列表
q_uid_list = [value['q_uid'] for value in data.values()]

# 将 q_uid 列表存储到新的 JSON 文件中
output_file_path = '/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/egoschema/subsub_video_names2.json'
with open(output_file_path, 'w') as outfile:
    json.dump(q_uid_list, outfile, indent=4)

print(f"q_uid 列表已存储到 {output_file_path}")