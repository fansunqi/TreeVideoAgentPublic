import json
import re
import os

def parse_json(text, video_id=None):
    # Add more robust error handling
    if text is None:
        print(f"{video_id}: No valid JSON found in the text {text}")
        return None
    
    text=re.sub('[\x00-\x09|\x0b-\x0c|\x0e-\x1f]','',text)
    
    # 尝试直接解析清理后的文本为 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 如果直接解析失败，使用正则表达式提取 JSON 对象和数组
    json_pattern = r"\{.*?\}|\[.*?\]"
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            match = match.replace("'", '"')
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    print(f"{video_id}: No valid JSON found in the text {text}")
    return None

def process_files(input_dir, json_output_dir, txt_output_dir):
    # 确保输出目录存在
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(txt_output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as txt_file:
                text = txt_file.read().decode('utf-8', errors='ignore')
            
            parsed_json = parse_json(text, video_id=filename)
            
            if parsed_json:
                json_output_path = os.path.join(json_output_dir, f'{filename}.json')
                with open(json_output_path, 'w', encoding='utf-8', errors='ignore') as json_file:
                    json.dump(parsed_json, json_file, ensure_ascii=False, indent=4)
                print(f"JSON data saved to {json_output_path}")
            else:
                txt_output_path = os.path.join(txt_output_dir, filename)
                with open(txt_output_path, 'w', encoding='utf-8', errors='ignore') as text_file:
                    text_file.write(text)
                print(f"Text data saved to {txt_output_path}")

# 输入目录和输出目录
input_dir = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs copy"
json_output_dir = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/parse_succ"
txt_output_dir = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/parse_fail"

# 处理文件
process_files(input_dir, json_output_dir, txt_output_dir)