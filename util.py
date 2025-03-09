import re
import pickle
import json
from pathlib import Path
from pprint import pprint
import os
import numpy as np
import random
# import torch
import logging
import pdb
from arg_parser import parse_args

args = parse_args()
cache_path = args.cache_path

# 检查 cache_path 文件是否存在
if os.path.exists(cache_path):
    cache_llm = pickle.load(open(cache_path, "rb"))
else:
    cache_llm = {}

### VideoTree ###
def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_fewshot_examples(qa_path, data_path):
    if len(qa_path) == 0 or len(data_path) == 0:
        return None
    qa = load_json(qa_path)
    data = load_json(data_path)  # uid --> str or list 
    examplars = []
    int_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    for i, (uid, examplar) in enumerate(qa.items()):
        description = data[uid]
        if isinstance(description, list):
            description = '. '.join(description)
        examplars.append(f"Examplar {i}.\n Descriptions: {description}.\n Question: {examplar['question']}\n A: {examplar['0']}\n B: {examplar['1']}\n C: {examplar['2']}\n D: {examplar['3']}\n E: {examplar['4']}\n Answer: {int_to_letter[examplar['truth']]}.")
    examplars = '\n\n'.join(examplars)
    return examplars

### VideoTree ###

### Mine ###
def get_video_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            filenames.append(os.path.splitext(filename)[0])
    return filenames


def get_intersection(list_a, list_b):
    return list(set(list_a) & set(list_b))

# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # 如果使用了 PyTorch 或 TensorFlow 等库，也需要设置它们的随机种子
    # torch.manual_seed(seed)
    # if you are using GPU
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # import tensorflow as tf
    # tf.random.set_seed(seed)

def get_frames_descriptions(parsed_candidate_descriptions):
    if parsed_candidate_descriptions == None:
        return None
    
    # 如果 parsed_candidate_descriptions 是列表，则取出第0个元素
    if isinstance(parsed_candidate_descriptions, list) and len(parsed_candidate_descriptions) > 0:
        parsed_candidate_descriptions = parsed_candidate_descriptions[0]
    
    # 如果 parsed_candidate_descriptions 不是字典，则返回 None
    if not isinstance(parsed_candidate_descriptions, dict):
        return None

    if "frame_descriptions" in parsed_candidate_descriptions:
        frames_descriptions = parsed_candidate_descriptions["frame_descriptions"]
        return frames_descriptions
    # elif "descriptions" in parsed_candidate_descriptions:
    #     frames_descriptions = parsed_candidate_descriptions["descriptions"]
    # elif "description" in parsed_candidate_descriptions:
    #     frames_descriptions = parsed_candidate_descriptions["description"]
    else:
        print(f"\nERROR --util.get_frames_descriptions--: {parsed_candidate_descriptions}\n")
        # if parsed_candidate_descriptions == None:
        #     print("\nparsed_candidate_descriptions is None\n")
        # raise KeyError
        return None


def parse_json(text, video_id=None):
    # TODO Add more robust error handling
    if text == None:
        print(f"{video_id}: No valid JSON found in the text {text}")
        return None
        # 再运行一遍，看一下这个问题有没有消失
    
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


def get_segment_id(description):
    for key in description.keys():
        if key.lower() == "segment_id":
            return int(description[key])
    return None


def get_duration(description):
    for key in description.keys():
        if key.lower() == "duration":
            return description[key]
    return None


def get_value_from_dict(d):
    if isinstance(d, dict) and len(d) == 1:
        key, value = next(iter(d.items()))
        return value
    return None


# Add more robust error handling
def parse_text_find_number(text, logger):
    item = parse_json(text)
    try:
        match = int(get_value_from_dict(item))
        if match in range(-1, 5):
            return match
        else:
            return random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        # pdb.set_trace()
        return -1

def print_nested_list(nested_list):
    print("\n")
    for one_list in nested_list:
        print(one_list)
    print("\n") 


def print_segment_list(video_segments):
    for seg in video_segments:
        print(f"[{seg.start} - {seg.end}] ", end='')
    print() 

### VideoAgent - stf ###
def parse_text_find_confidence(text, logger):
    item = parse_json(text)
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        else:
            # return random.randint(1, 3)
            return 1
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1

def read_caption(captions, sample_idx):
    video_caption = {}
    for idx in sample_idx:
        video_caption[f"frame {idx}"] = captions[idx - 1]
    return video_caption

def get_from_cache(key, logger=None, use_logger=True):
    try:
        return cache_llm[key.encode()].decode()
    except KeyError:
        pass
    except Exception as e:
        if use_logger:
            logger.warning(f"Error getting from cache: {e}")
    return None


def save_to_cache(key, value, logger=None, use_logger=True):
    try:
        cache_llm[key.encode()] = value.encode()
        pickle.dump(cache_llm, open(cache_path, "wb"))
    except Exception as e:
        if use_logger:
            logger.warning(f"Error saving to cache: {e}")


def set_logger(timestamp, logger_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger_file_path = os.path.join(logger_path, f"ta_subset_{timestamp}.log")
    file_handler = logging.FileHandler(logger_file_path)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
    )
    file_handler.setFormatter(formatter)
   
    # 移除终端输出
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(file_handler)
    
    return logger


if __name__ == "__main__":


    duration = '23-45'
    start, end = map(int, duration.split('-'))
    print(start, end)

    

    # candiate_descriptions = '''
    # {
    #   "frame_descriptions": [
    #     {
    #       "segment_id": "1",
    #       "duration": "1 - 23",
    #       "description": "Frame 1 shows C checking the knife, indicating the beginning of the knife handling process."
    #     },
    #     {
    #       "segment_id": "2",
    #       "duration": "23 - 45",
    #       "description": "Frame 23 shows C pulling the edge of the grill, possibly in preparation for sharpening or cutting, suggesting the setup phase."
    #     },
    #     {
    #       "segment_id": "3",
    #       "duration": "45 - 67",
    #       "description": "Frame 45 shows C cutting the protective covering, hinting at C’s attention to detail and care in the process."
    #     },
    #     {
    #       "segment_id": "4",
    #       "duration": "67 - 90",
    #       "description": "Frame 67 shows C cutting the wood, indicating the progression of their cutting technique and potential sharpening improvements."
    #     },
    #     {
    #       "segment_id": "5",
    #       "duration": "90 - 112",
    #       "description": "Frame 90 shows C cutting the food, which may signify the beginning of a more refined knife handling technique after initial sharpening."
    #     },
    #     {
    #       "segment_id": "6",
    #       "duration": "112 - 135",
    #       "description": "Frame 112 shows C removing a piece of wood with their hand, indicating hands-on adjustments during the knife use, potentially refining their approach."
    #     },
    #     {
    #       "segment_id": "7",
    #       "duration": "135 - 157",
    #       "description": "Frame 135 shows C cutting the wood with a blade, suggesting C is now more focused and precise with their technique."
    #     },
    #     {
    #       "segment_id": "8",
    #       "duration": "157 - 180",
    #       "description": "Frame 157 shows C putting down a cloth, indicating a moment of rest or preparation, which might signal a change in how they approach the knife’s sharpness."
    #     }
    #   ]
    # }
    # '''
    # print(parse_json(candiate_descriptions))
    
    