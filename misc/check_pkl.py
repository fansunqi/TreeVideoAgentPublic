import pickle
import json
import re
import os
from pprint import pprint
import pdb

cache_path = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/tva_cache_gpt4_no_cheat copy.pkl"
txt_path = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/tva_cache_gpt4_no_cheat copy.txt"
json_output_dir = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs"

# 确保输出目录存在
os.makedirs(json_output_dir, exist_ok=True)

def parse_json(text):
    # 匹配整个字符串
    pattern = r'\["gpt.*?\]\]'
    
    # 使用正则表达式查找匹配项
    matches = re.findall(pattern, text, re.DOTALL)

    for i, match in enumerate(matches):
        pprint(match)
        try:
            parsed_json = json.loads(match)  # 直接解析整个匹配项
            pprint(parsed_json)
            
            # 将每个解析后的 JSON 对象存储到单独的 JSON 文件中
            json_output_path = os.path.join(json_output_dir, f'parsed_json_{i}.json')
            with open(json_output_path, 'w', encoding='utf-8') as json_file:
                json.dump(parsed_json, json_file, ensure_ascii=False, indent=4)
            print(f"JSON data saved to {json_output_path}")
            
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
        pdb.set_trace()

# 读取文件中的全部字节并存储到文本文件
with open(cache_path, 'rb') as f:
    chunk = f.read()  # 读取文件中的全部字节
    # print(chunk[:1000])  # 输出查看文件内容

    # 将读取的内容存储到文本文件
    with open(txt_path, 'wb') as txt_file:
        txt_file.write(chunk)

# print("txt saving done")

# 从文本文件中读取数据
# with open(txt_path, 'rb') as txt_file:
#     text = txt_file.read().decode('utf-8')  # 读取文本文件内容并解码为字符串
#     print(text[:2000])
#     parse_json(text)  # 解析读取的内容


