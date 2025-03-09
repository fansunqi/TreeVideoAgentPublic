import json
import os
from tqdm import tqdm

# 读取JSON文件
with open('/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/egoschema/subset_anno.json', 'r') as file:
    data = json.load(file)

# 创建下载目录
download_dir = ''
os.makedirs(download_dir, exist_ok=True)

# 遍历JSON数据并下载文件
for key, value in tqdm(data.items(), desc="Downloading files"):
    q_uid = value.get('q_uid')
    google_drive_id = value.get('google_drive_id')
    if google_drive_id:
        file_path = os.path.join(download_dir, f'{q_uid}.mp4')
        if not os.path.exists(file_path):
            try:
                os.system(f'gdown --id {google_drive_id} -O {file_path}')
            except Exception as e:
                print(f"Failed to download file with Google Drive ID {google_drive_id}: {e}")
        else:
            print(f"File {file_path} already exists, skipping download.")

print("下载完成")