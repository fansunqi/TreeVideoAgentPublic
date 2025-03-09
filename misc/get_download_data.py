import json
import os
from tqdm import tqdm

# 读取JSON文件
with open('/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/egoschema/subset_anno.json', 'r') as file:
    data = json.load(file)

# 创建下载目录
download_dir = '/Users/sunqifan/Documents/codes/video_agents/EgoSchema/videos'
os.makedirs(download_dir, exist_ok=True)

# 存储已成功下载的视频信息
downloaded_videos = {}

# 遍历JSON数据并下载文件
for key, value in tqdm(data.items(), desc="Downloading files"):
    q_uid = value.get('q_uid')
    google_drive_id = value.get('google_drive_id')
    if google_drive_id:
        file_path = os.path.join(download_dir, f'{q_uid}.mp4')
        if os.path.exists(file_path):
            downloaded_videos[q_uid] = value


# 将已成功下载的视频信息存储到 JSON 文件
with open('/Users/sunqifan/Documents/codes/video_agents/EgoSchema/downloaded_videos.json', 'w') as outfile:
    json.dump(downloaded_videos, outfile, indent=4)

print("下载完成")