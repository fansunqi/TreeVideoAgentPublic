import cv2
import os

def capture_frames_from_video(video_path, output_dir, interval):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_count = 0
    saved_count = 0
    
    while frame_count < total_frames:
        print(f"Processing frame {frame_count}")
        
        # 设置视频帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        # 读取帧
        ret, frame = cap.read()
        
        if ret:
            # 保存帧为图片
            output_image_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_image_path, frame)
            print(f"Frame {frame_count} saved as {output_image_path}")
            saved_count += 1
        else:
            print(f"Failed to capture frame at {frame_count}")
        
        # 增加帧计数器
        frame_count += interval
    
    # 释放视频捕获对象
    cap.release()
    print(f"Total {saved_count} frames saved.")

# 示例用法
interval = 500
video_path = '/Users/sunqifan/Desktop/egoschema_videos/subset_videos/0a8b2c9d-b54c-4811-acf3-5977895d2445.mp4'
# 获取路径中的最后一个文件名
file_name = os.path.splitext(os.path.basename(video_path))[0]
output_dir = os.path.join('/Users/sunqifan/Desktop/egoschema_videos/output_frames', f'{interval}_{file_name}')
capture_frames_from_video(video_path, output_dir, interval)

