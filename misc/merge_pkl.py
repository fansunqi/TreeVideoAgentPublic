import pickle
import os
import json

new_cache_path = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/cache_gpt4_try2.pkl"
cache_llm = {}

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def save_to_cache(key, value, logger=None, use_logger=True):
    try:
        cache_llm[key.encode()] = value.encode()
        pickle.dump(cache_llm, open(new_cache_path, "wb"))
    except Exception as e:
        if use_logger:
            logger.warning(f"Error saving to cache: {e}")

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
key_path = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs_try1/key/"
value_path1 = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs_try1/parsed_value/"
value_path2 = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/cache/egoschema/json_outputs_try1/reprocess_succ_unparsed_value"


for filename in os.listdir(key_path):
    key_file_path = os.path.join(key_path, filename)
    # 在 value_path1 和 value_path2 下面寻找同名文件
    value_file_path1 = os.path.join(value_path1, filename)
    value_file_path2 = os.path.join(value_path2, filename)
    
    value_data = None
    if os.path.exists(value_file_path1):
        value_data = load_json_file(value_file_path1)
    elif os.path.exists(value_file_path2):
        value_data = load_json_file(value_file_path2)
    
    if value_data is not None:
        key_data = load_json_file(key_file_path)
        cache_llm[str(key_data).encode()] = str(value_data).encode()

print(len(cache_llm))
pickle.dump(cache_llm, open(new_cache_path, "wb"))

        





