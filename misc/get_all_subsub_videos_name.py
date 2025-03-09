import os
import json

def get_mp4_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            filenames.append(os.path.splitext(filename)[0])
    return filenames

def save_to_json(data, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == '__main__':
    # 指定目录路径
    directory_path = '/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/egoschema/subsub_videos'
    # 指定输出JSON文件路径
    output_json_path = '/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/egoschema/subsub_video_names.json'

    # 获取所有.mp4文件的文件名（去掉.mp4后缀）
    mp4_filenames = get_mp4_filenames(directory_path)

    # 将结果存储为JSON文件
    save_to_json(mp4_filenames, output_json_path)

    print(f"File names saved to {output_json_path}")