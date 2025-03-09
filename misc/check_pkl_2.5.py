import json
import re

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
    # if text is None:
    #     print(f"{video_id}: No valid JSON found in the text {text}")
    #     return None
    
    # 去除非 JSON 数据的二进制字符
    # text = re.sub('[\x00-\x09|\x0b-\x0c|\x0e-\x1f]', '', text)
    # text = strip_control_characters(text)
    text = remove_invalid_chars(text)

    
    
    # 尝试直接解析清理后的文本为 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # print("text\n", text)

    json_pattern = r"\{.*?\}|\[.*?\]"
    # json_pattern = r"\{.*\}|\[.*\]"

    matches = re.findall(json_pattern, text, re.DOTALL)
    # print("\nmatches\n", matches)
    # print("\nmatches[0]\n", repr(matches[0]))
    # print("\nover!")

    # with open('output.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(repr(matches[0]), json_file, ensure_ascii=False, indent=4)
    # print("JSON data saved to output.json")


    for match in matches:
        # try:
        # print("match", match)
        # match = match.replace("'", '"')
        return json.loads(match)
        # except json.JSONDecodeError:
        #     continue
    
    # # print(f"{video_id}: No valid JSON found in the text {text}")
    # return None

test_txt = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs_try1/reprocess_fail_unparsed_value/445.txt"
with open(test_txt, "rb") as txt_file:
    text = txt_file.read().decode('utf-8', errors='ignore')

parsed_json = parse_json(text)
print(parsed_json)