import json
import re
import os

succ_num = 0
fail_num = 0

def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def remove_invalid_chars(text):
   return ''.join(c for c in text if c.isprintable())

def strip_control_characters(s):
    word = ''
    for i in s:
        if ord(i)>31 or ord(i) == 10 or ord(i) ==13:
            word += i
    return word

import ftfy
def remove_invalid_chars(text):
    return ftfy.fix_text(text)

def parse_json(text, video_id=None):
    # Add more robust error handling
    if text is None:
        print(f"{video_id}: No valid JSON found in the text {text}")
        return None

    
    # 尝试直接解析清理后的文本为 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # print("text\n", text)
    # json_pattern = r"\{.*?\}|\[.*?\]"
    json_pattern = r"\{.*\}|\[.*\]"
    matches = re.findall(json_pattern, text, re.DOTALL)
    if len(matches) == 0:
        json_pattern_2 = r"\{.*?\}|\[.*?\]"
        matches = re.findall(json_pattern_2, text, re.DOTALL)
        if len(matches) == 0:
            return None

    for match in matches:
        try:
            # match = match.replace("'", '"')  # 一定要去掉
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None

# 指定目录
directory = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs_try1/unparsed_value"
output_directory = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs_try1/reprocess_succ_unparsed_value"
fail_output_directory = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs_try1/reprocess_fail_unparsed_value"

# 获取所有文件路径
all_file_paths = get_all_file_paths(directory)

for filepath in all_file_paths:
    with open(filepath, "rb") as txt_file:
        text = txt_file.read().decode('utf-8', errors='ignore')

    # 去除二进制字符
    # text = re.sub('[\x00-\x09|\x0b-\x0c|\x0e-\x1f]', '', text)
    # text = strip_control_characters(text)
    text = remove_invalid_chars(text)

    parsed_json = parse_json(text)
    
    if parsed_json != None:
        # 获取文件名
        filename = os.path.basename(filepath).replace('.txt', '.json')
        # 构建输出文件路径
        output_filepath = os.path.join(output_directory, filename)
        # 将 parsed_json 写入文件
        with open(output_filepath, 'w', encoding='utf-8') as json_file:
            json.dump(parsed_json, json_file, ensure_ascii=False, indent=4)
        print(f"JSON data saved to {output_filepath}")

        succ_num += 1
    else:
        # 获取文件名
        filename = os.path.basename(filepath)
        # 构建失败输出文件路径
        fail_output_filepath = os.path.join(fail_output_directory, filename)
        # 将原文件内容写入失败目录
        with open(fail_output_filepath, 'w', encoding='utf-8') as fail_file:
            fail_file.write(text)
        print(f"Failed to parse JSON, original data saved to {fail_output_filepath}")
        fail_num += 1




print(succ_num)