import argparse
import json
import os
from together import Together
import base64
import pandas as pd
import cv2
import os


def sample_frames_from_video(video_path, samples_num, output_dir=None):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算采样间隔
    interval = total_frames // samples_num
    
    frame_count = 0
    sampled_frames = []
    
    for i in range(samples_num):
        # 设置视频帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        # 读取帧
        ret, frame = cap.read()
        
        if ret:
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_image_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(output_image_path, frame)
                print(f"Frame {frame_count} saved as {output_image_path}")
                sampled_frames.append(encode_image(output_image_path))
            else:
                # 临时保存帧到内存中以便编码
                _, buffer = cv2.imencode('.jpg', frame)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                sampled_frames.append(encoded_image)
        else:
            print(f"Failed to capture frame at {frame_count}")
        
        # 增加帧计数器
        frame_count += interval
    
    # 释放视频捕获对象
    cap.release()
    
    return sampled_frames


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def evaluate_model_on_egoschema(model, qa_path, data_path, video_dir, output_file, samples_num):
    
    # 读取 Parquet 文件
    qa_df = pd.read_parquet(qa_path)
    qa = qa_df.to_dict(orient='index')

    video_idx_list = ["0a8b2c9d-b54c-4811-acf3-5977895d2445"]
    results = []

    for video_idx in video_idx_list:
        
        video_path = os.path.join(video_dir, f"{video_idx}.mp4")
        base64_sampled_frames = sample_frames_from_video(video_path, samples_num)

        prompt = f"Descriptions: {description}.\nQuestion: {examplar['question']}\nA: {examplar['0']}\nB: {examplar['1']}\nC: {examplar['2']}\nD: {examplar['3']}\nE: {examplar['4']}\n"

        stream = model.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            stream=True,
        )

        response = ""
        for chunk in stream:
            response += chunk.choices[0].delta.content or ""

        results.append({
            "uid": uid,
            "question": examplar['question'],
            "predicted_answer": response.strip(),
            "true_answer": examplar['truth']
        })

    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"Evaluation results saved to {output_file}")



def main():
    parser = argparse.ArgumentParser(description="Evaluate MLLM on EgoSchema dataset")
    parser.add_argument("--qa_path", required=True, type=str, help="Path to the QA JSON file")
    parser.add_argument("--data_path", required=True, type=str, help="Path to the data JSON file")
    parser.add_argument("--image_dir", required=True, type=str, help="Directory containing the images")
    parser.add_argument("--output_file", required=True, type=str, help="Path to the output Excel file")
    args = parser.parse_args()

    client = Together()
    evaluate_model_on_egoschema(client, args.qa_path, args.data_path, args.image_dir, args.output_file)



if __name__ == "__main__":
    # main()

    # sample_frames_from_video()示例用法
    video_path = '/Users/sunqifan/Desktop/egoschema_videos/subset_videos/0a8b2c9d-b54c-4811-acf3-5977895d2445.mp4'
    samples_num = 20
    output_dir = video_path.replace('.mp4','_frames')
    sampled_frames = sample_frames_from_video(video_path, samples_num, output_dir)
    # 打印采样的帧数
    print(f"Total {len(sampled_frames)} frames sampled.")