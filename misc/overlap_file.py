import os

def get_filenames(directory):
    """获取指定目录中的所有文件名（不包括路径）"""
    return set(os.listdir(directory))

def find_common_files(dir1, dir2):
    """查找两个目录中重复的文件名"""
    files1 = get_filenames(dir1)
    files2 = get_filenames(dir2)
    common_files = files1.intersection(files2)
    return common_files

# 指定两个目录
dir1 = '/Users/sunqifan/Documents/codes/video_agents/EgoSchema/videos'
dir2 = '/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/egoschema/subsub_videos'

# 查找重复的文件名
common_files = find_common_files(dir1, dir2)

# 打印重复的文件名及数量
print(f"重复的文件数量: {len(common_files)}")
print("重复的文件名:")
for filename in common_files:
    print(filename)