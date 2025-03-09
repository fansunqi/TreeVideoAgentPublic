import json
import re
import os
from pprint import pprint
import pdb
from tqdm import tqdm
import pickle

cache_path = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/tva_cache_gpt4_no_cheat copy.pkl"
txt_path = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/tva_cache_gpt4_no_cheat copy.txt"
json_output_dir = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs copy"

new_cache_path = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/cache_gpt4_new.pkl"
cache_llm = {}

succ_num = 0
fail_num = 0
# 确保输出目录存在
os.makedirs(json_output_dir, exist_ok=True)

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def save_to_cache(key, value, logger=None, use_logger=True):
    try:
        cache_llm[key.encode()] = value.encode()
        pickle.dump(cache_llm, open(cache_path, "wb"))
    except Exception as e:
        if use_logger:
            logger.warning(f"Error saving to cache: {e}")


def parse_json(text, video_id=None):
    # Add more robust error handling
    if text is None:
        print(f"{video_id}: No valid JSON found in the text {text}")
        return None
    
    # 先检查是不是 ```json\n{}\n```的格式
    pattern = r'```json\n({.*?})\n```'
    
    # 使用正则表达式查找匹配项
    match = re.search(pattern, text, re.DOTALL)

    if match:
        text = match.group(1)

    try:
        # First, try to directly parse the text as JSON
        return json.loads(text)
    
    except json.JSONDecodeError:
        # If direct parsing fails, use regex to extract JSON

        # Pattern for JSON objects and arrays
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

def process_between_content(between_content, i):
    global succ_num
    global fail_num
    parsed_between_content = parse_json(between_content)
    if parsed_between_content:
        succ_num += 1
        # 将解析后的 JSON 对象存储到单独的 JSON 文件中
        # between_content_json_path = os.path.join(json_output_dir, f'{i}_parsed_answer.json')
        # with open(between_content_json_path, 'w', encoding='utf-8', errors='ignore') as json_file:
        #     json.dump(parsed_between_content, json_file, ensure_ascii=False, indent=4)
        # print(f"Between content JSON data saved to {between_content_json_path}")

    else:
        fail_num += 1
        # 将匹配项之间的内容存储到单独的文本文件中
        # between_content_txt_path = os.path.join(json_output_dir, f'unparsed_{i}_answer.txt')
        # with open(between_content_txt_path, 'w', encoding='utf-8', errors='ignore') as text_file:
        #     text_file.write(between_content)
        # print(f"Between content saved to {between_content_txt_path}")
    return parsed_between_content

def parse_and_store_json(text):
    # 匹配整个字符串，包括匹配项之间的内容
    pattern = r'(\["gpt.*?\]\])(.*?)(?=\["gpt|\Z)'
    
    # 使用正则表达式查找匹配项
    matches = re.findall(pattern, text, re.DOTALL)

    for i, (match, between_content) in tqdm(enumerate(matches)):
        
        try:
            key = str(json.loads(match))  # 直接解析整个匹配项
            
            # 将每个解析后的 JSON 对象存储到单独的 JSON 文件中
            # json_output_path = os.path.join(json_output_dir, f'{i}_parsed_question.json')
            # with open(json_output_path, 'w', encoding='utf-8', errors='ignore') as json_file:
            #     json.dump(parsed_json, json_file, ensure_ascii=False, indent=4)
            # print(f"JSON data saved to {json_output_path}")
            
            # 处理匹配项之间的内容
            value = str(process_between_content(between_content, i))

            cache_llm[key.encode()] = value.encode()
            
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

        # if i > 200:
        #     break

# 从文本文件中读取数据
with open(txt_path, 'rb') as txt_file:
    text = txt_file.read().decode('utf-8', errors='ignore')  # 读取文本文件内容并解码为字符串
    parse_and_store_json(text)  # 解析读取的内容

pickle.dump(cache_llm, open(new_cache_path, "wb"))

print(succ_num)
print(fail_num)

