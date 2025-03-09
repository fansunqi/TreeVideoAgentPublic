import cv2
from pathlib import Path
from tqdm import tqdm
import json


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def extract_es():
    input_base_path = Path('data/egoschema/subsub_videos')
    output_base_path = Path('data/egoschema/subsub_frames')
    fps = 1
    pbar = tqdm(total=len(list(input_base_path.iterdir())))
    for video_fp in input_base_path.iterdir():
        output_path = output_base_path / video_fp.stem
        output_path.mkdir(parents=True, exist_ok=True)
        vidcap = cv2.VideoCapture(str(video_fp))
        count = 0
        frame_count = 0  # 用于保存帧的编号
        success = True
        fps_ori = int(vidcap.get(cv2.CAP_PROP_FPS))   
        frame_interval = int(1 / fps * fps_ori)
        while success:
            success, image = vidcap.read()
            if not success:
                break
            if count % frame_interval == 0:
                cv2.imwrite(f'{output_path}/{frame_count}.jpg', image)
                frame_count += 1
            count += 1
        pbar.update(1)
    pbar.close()


def extract_frames_from_video(video_fp, output_dir, fps=1):
    """
    从单个视频文件中提取帧并保存到指定目录。

    参数:
    video_fp (str or Path): 视频文件路径。
    output_dir (str or Path): 保存提取帧的目录。
    fps (int): 每秒提取的帧数。
    """
    video_fp = Path(video_fp)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vidcap = cv2.VideoCapture(str(video_fp))
    count = 0
    frame_count = 0  # 用于保存帧的编号
    success = True
    fps_ori = int(vidcap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(1 / fps * fps_ori)

    while success:
        success, image = vidcap.read()
        if not success:
            break
        if count % frame_interval == 0:
            cv2.imwrite(f'{output_dir}/{frame_count}.jpg', image)
            frame_count += 1
        count += 1

    print(f"提取完成: {video_fp} -> {output_dir}")


if __name__ == '__main__':
    # extract_es()

    video_file_path = '/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/longvideobench/example/banner_video.mp4'  # 替换为实际的视频文件路径
    output_directory = '/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/data/longvideobench/example/frames'  # 替换为实际的输出目录
    extract_frames_from_video(video_file_path, output_directory, fps=1)