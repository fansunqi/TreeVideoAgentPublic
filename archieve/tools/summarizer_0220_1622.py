# 总结者 based on https://github.com/Hyu-Zhang/HCQA/
# 总结各帧描述，感受时序变化
import os
import re
import json
import time
import pdb
import ast
import openai
from tqdm import tqdm
from model import GPT
import numpy as np
from datetime import datetime
from util import read_caption, parse_json


systerm = '''
You're a visual summary expert. You can accurately make a [SUMMARY] based on [CAPTION], where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.
'''

incontext_prompt = '''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each caption represents a frame in the video in chronological order, although they may not be consecutive. And each description is separated by a newline character. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
[SUMMARY]: Based on the CAPTIONS of these video clips, you need to summarise them into an overall description of the video, in chronological order.
I will give you an example as follow:
<Example>
{example}
Now, you should make a [SUMMARY] based on the [CAPTION] below. You SHOULD follow the format of example.
[CAPTION]
{caption}
[SUMMARY]
'''

systerm_qa='''
You are a visual question answering expert. You can choose the correct answer from five options of [OPTION] based on the [CAPTION], [SUMMARY], [QUESTION], and [REASON]. Where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.
'''

incontext_prompt_qa='''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each caption represents a frame in the video in chronological order, although they may not be consecutive. And each description is separated by a newline character. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
[SUMMARY]: Based on the CAPTIONS of these video clips, an overall description of the video, in chronological order.
[QUESTION]: A question about video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [SUMMARY], reasoning step by step to get the answer. If [SUMMARY] doesn't have enough information, you need to get it from the [CAPTION].
I will give you some examples as follow:
{example}
Now, you should first make a [REASON] based on [QUESTION] and [SUMMARY], then give right number of [OPTION] as [ANSWER] . Additionally, you need to give me [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You SHOULD answer the question, even given a low confidence.
[CAPTION]
{caption}
[SUMMARY]
{summary}
[QUESTION]
{question}
[OPTION]
{option}
'''

response_prompt='''
YOU MUST output in the JSON format.
{
"REASON":"[REASON]",
"ANSWER":"[ANSWER]",
"CONFIDENCE": "[CONFIDENCE]"
}
'''


example_summary_path = "data/egoschema/example_summary.txt"
with open(example_summary_path,'r') as ex:
    example_summary = ex.read()

example_qa_by_summary_path = "data/egoschema/example_qa_by_summary.txt"
with open(example_qa_by_summary_path,'r') as ex:
    example_qa_by_summary = ex.read()

# TODO 之后 summarizer 放到 main.py 中创建
api_key = "sk-1KQu8Ow6E8bndCHLBbC06f484dCd47CcBe2342C8B5E8C9B2"
# api_key = "sk-XiMnVWdBVk02Iyv160E9B5Ee3fA649BfAd4b442074B6B84a"
base_url = "https://api.juheai.top/v1/"
model_name = "gpt-4o"
summarizer = GPT(api_key=api_key, model_name=model_name, base_url=base_url)
qa_model_by_summary = GPT(api_key=api_key, model_name=model_name, base_url=base_url)

def HCQA_main(exmaple):
    # 从 model.py 中引入 gpt
    api_key = "sk-1KQu8Ow6E8bndCHLBbC06f484dCd47CcBe2342C8B5E8C9B2"
    base_url = "https://api.juheai.top/v1/"
    model_name = "gpt-4o"
    summarizer = GPT(api_key=api_key, model_name=model_name, base_url=base_url)


    # 设置 system-level prompt 和用户级 prompt 列表
    # head = "You are a helpful assistant."
    # prompt = "What is the capital of France?"

    # # 调用 forward 方法获取响应
    # response, info = summarizer.forward(head, prompt)

    # # 打印响应和信息
    # print("Response:", response)
    # print("Info:", info)


    qa_list = os.listdir('/Users/sunqifan/Documents/codes/video_agents/HCQA/LaViLa_cap5_subset/')
    output_dir = '/Users/sunqifan/Documents/codes/video_agents/HCQA/summary/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    exist_files = os.listdir(output_dir)
    qa_list = [item for item in qa_list if item.replace('json','txt') not in exist_files]
    print(len(qa_list))

    num_tasks = 10

    queries = {}
    questions_path = "/Users/sunqifan/Documents/codes/video_agents/HCQA/questions.json"
    with open(questions_path,'r') as f1:
        data=json.load(f1)
    for item in data:
        queries[item['q_uid']] = item

    LaViLa_cap5_subset_path = "/Users/sunqifan/Documents/codes/video_agents/HCQA/LaViLa_cap5_subset/"

    for i in range(100):
        try:
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in qa_list if f.replace('json','txt') not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]

            # with Pool(10) as pool:
            #     pool.starmap(llm_inference, task_args)

            # 单线程版本
            # 单线程处理每个任务
            for part in all_parts:
                # llm_inference(*task_arg)  # 直接调用函数，替换pool.starmap
                for file in tqdm(part): 
                    uid = file[:-5]

                    try:
                        with open(LaViLa_cap5_subset_path + uid + '.json','r') as f:
                            captions=json.load(f)
                        caps = ''
                        for c in captions:
                            caps += c['Caption'] + "\n"
                        
                        # len(captions):45
                        # caps:
                        # '#C C ties the Napier grass; #C C ties the Napier grass ; #C C ties the Napier grass ; #C C ties the Napier grass ; #C C Ties papyrus reed with both\n#C C folds the Napier grass ; #C C ties the leaves with both hands; #C C Ties papyrus reed with both; #C C ties the leaflets together with both hands.; #C C ties the Napier grass \n#C C drops the twine on a pile of others on the floor; #C C Drops papyrus reed on the floor; #C C drops the frond on the floor with her right hand.; #C C drops the leaf on the floor; #C C drops the twine on a pile of others on the floor\n#C C Tears papyrus reed with cutting object; #C C Holds with both hands; #C C Ties papyrus reed with cutting; #C C Holds with both hands; #C C Tears papyrus reed with cutting object\n#C C trims leaflet from a stick with the sharp edge iron rod on the; #C C splits the cattail leaves; #C C cuts the midrib on the palm frond.; #C C splits the cattail leaves; #C C trims leaflet from a stick with the sharp edge iron rod on the\n#C C Picks papyrus reed from the; #C C Picks papyrus reed from the; #C C Picks papyrus reed from the; #C C Picks papyrus reed from the; #C C Picks papyrus reed from the\n#C C Tears papyrus reed with cutting object; #C C pulls the midrib on the leaflet; #C C pushes coconut leaves on a floor; #C C files the palm frond with the file with both hands.; #C C trims leaflet from a stick with the sharp edge iron rod on the\n#C C drops the Napier grass ; #C C drops the twine on a pile of others on the floor; #C C drops the Napier grass ; #C C Drops papyrus reed on the floor; #C C drops the twine on a pile of others on the floor\n#C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting\n#C C lifts the Napier grass ; #C C Tears papyrus reed with cutting object; #C C removes broom stick from the palm frond with right hand; #C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object\n#C C Pushes papyrus reeds a; #C C moves the Napier grass ; #C C Drops papyrus reed on the; #C C puts the Napier grass down ; #C C Drops papyrus reed on the\n#C C trims leaflet from a stick with the sharp edge iron rod on the; #C C removes the twine from the leaf; #C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object\n#C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C breaks the coconut leaf; #C C Tears papyrus reed with cutting; #C C tears the cattail leaves\n#C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C Ties papyrus reed with cutting\n#C C trims leaflet from a stick with the sharp edge iron rod on the floor; #C C trims leaflet from a stick with the sharp edge iron rod on the floor; #C C trims leaflet from a stick with the sharp edge iron rod on the floor; #C C trims leaflet from a stick with the sharp edge iron rod on the floor; #C C cuts sisal \n#C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C pulls the Napier grass ; #C C trims leaflet from a stick with the sharp edge iron rod on the floor; #C C Tears papyrus reed with cutting object\n#C C pushes Dracaena Marginata leaves on the floor; #C C drops the filed palm frond on the floor with right hand; #C C pulls the midrib on the palm frond.; #C C puts sisal on the floor; #C C Tears papyrus reed with cutting object\n#C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting\n#C C Ties the papyrus reed with cutting object; #C C Ties with cutting object; #C C removes broom stick from the palm frond with right hand; #C C Tears papyrus reed with cutting object; #C C Ties papyrus reed with cutting object\n#C C picks Napier grass ; #C C takes another leaf from the floor; #C C Holds with both hands; #C C Picks papyrus reed from the; #C C Holds with both hands\n#C C Tears papyrus reed with cutting object; #C C cuts the palm frond with both hands; #C C Tears papyrus reed with cutting object; #C C cuts the Napier grass; #C C Tears papyrus reed with cutting object\n#C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C cuts the Napier grass \n#C C Drops papyrus reed on the; #C C drops the Napier grass; #C C Drops papyrus reed on the; #C C drops the Napier grass ; #C C Drops papyrus reed on the\n#C C Tears papyrus reed with cutting; #C C cuts the Napier grass ; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting\n#C C Drops papyrus reed on the floor; #C C Pushes papyrus reed ; #C C Holds with both hands; #C C Pushes papyrus reed ; #C C Pushes papyrus reed  with both\n#C C Picks papyrus reed from the floor; #C C Ties papyrus reed with cutting; #C C Ties papyrus reed with cutting; #C C Ties papyrus reed with cutting; #C C Ties papyrus reed with cutting\n#C C Tears papyrus reed with cutting; #C C Picks papyrus reed from the; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting\n#C C Ties papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C Ties the papyrus reed with both\n#C C Holds with both hands; #C C Picks papyrus reed from the floor; #C C Tears papyrus reed with cutting object; #C C picks a palm frond from the floor with her right hand; #C C picks the Napier grass \n#C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Ties with cutting object; #C C Tears papyrus reed with cutting\n#C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C tears the cattail leaves\n#C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C splits the cattail leaves; #C C holds the cattail leaves\n#C C Tears papyrus reed with cutting object; #C C files the palm frond with the file with both hands.; #C C files the palm frond with the file with both hands.; #C C trims leaflet from a stick with the sharp edge iron rod on the; #C C splits the cattail leaves\n#C C lifts Napier grass ; #C C Picks papyrus reed from the; #C C drops the Napier grass on the; #C C lifts the Napier grass ; #C C Picks papyrus reed from the\n#C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C removes the twine from the leaf\n#C C Drops on the floor; #C C Picks papyrus reed from the; #C C Drops papyrus reed on the; #C C drops the Napier grass; #C C Drops papyrus reed on the\n#C C Ties papyrus reed with cutting; #C C drops the Napier grass ; #C C Tears papyrus reed with cutting object; #C C Ties papyrus reed with cutting; #C C Tears papyrus reed with cutting object\n#C C Drops papyrus reed on the floor; #C C drops the twine on a pile of others on the floor; #C C Puts down  papyrus reed; #C C Drops papyrus reed on the floor; #C C drops the Napier grass \n#C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting\n#C C Tears papyrus reed with cutting object; #C C Ties the papyrus reed with cutting; #C C Tears papyrus reed with cutting object; #C C Ties with cutting object; #C C Ties with cutting object\n#C C picks the Napier grass ; #C C Picks papyrus reed from the; #C C Picks papyrus reed from the; #C C picks the Napier grass ; #C C Picks papyrus reed from the\n#C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C Tears papyrus reed with cutting object; #C C cuts the palm frond with the pair of scissors in both hands\n#C C drops the twine on a pile of others on the floor; #C C drops the twine on a pile of others on the floor; #C C lifts the Napier grass ; #C C drops the twine on a pile of others on the floor; #C C lifts the Napier grass \n#C C Tears papyrus reed with cutting; #C C cuts the Napier grass ; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting\n#C C Ties with cutting object; #C C Tears papyrus reed with cutting; #C C lifts the Napier grass ; #C C Tears papyrus reed with cutting; #C C Tears papyrus reed with cutting\n'

                        instruction = str(uid) + "\n" + systerm + "\n" + incontext_prompt.format(example = example_summary, caption = caps)
                        response, info = summarizer.forward(head = None, prompt = instruction)
                        with open(f"{output_dir}/{uid}.txt", "w") as f:
                            f.write(response)

                    except Exception as e:
                        print(e)
                        print(f"Error occurs when processing this query: {uid}", flush=True)
                        break         
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")


def summarize_lavia_subset(summarizer, example, all_cap_file, output_dir, interval):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_caps = json.load(open(all_cap_file, "r"))
    for video_id, captions in tqdm(all_caps.items()):
        num_frames = len(captions) # 180
        caps = ''
        for c_id, c in enumerate(captions):
            if c_id % interval == 0:
                caps += c + "\n"
        instruction = str(video_id) + "\n" + systerm + "\n" + incontext_prompt.format(example=example, caption=caps)
        response, info = summarizer.forward(head=None, prompt=instruction)
        with open(f"{output_dir}/{video_id}.txt", "w") as f:
            f.write(response)


def qa_lavia_subset(summary_path, output_dir):
    qa_list = os.listdir(summary_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 忽略已存在的文件，便于重启
    exist_files = os.listdir(output_dir)
    qa_list = [item for item in qa_list if item not in exist_files]
    print(len(qa_list))

    subset_anno_path = "data/egoschema/subset_anno.json"
    with open(subset_anno_path,'r') as f1:
        subset_anno = json.load(f1)

    all_cap_file = "data/egoschema/lavila_subset.json"
    all_caps = json.load(open(all_cap_file, "r"))

    for file in tqdm(qa_list):
        # pdb.set_trace()
        uid = file[:-4]
        d = subset_anno[uid]
        try:
            # 区别是在于这里用了所有 captions
            # 而不是截取的 captions
            captions = all_caps[uid]
            caps = ''
            for c in captions:
                caps += c + "\n"
            with open(os.path.join(summary_path, f'{uid}.txt'), 'r') as f1:
                sum = f1.read()
            opt = ''
            que = 'question: ' + d['question']
            opt += 'option 0: ' + d['option 0'] + "\n"
            opt += 'option 1: ' + d['option 1'] + "\n"
            opt += 'option 2: ' + d['option 2'] + "\n"
            opt += 'option 3: ' + d['option 3'] + "\n"
            opt += 'option 4: ' + d['option 4'] + "\n"

            instruction = str(uid) + "\n" + systerm_qa + "\n" + incontext_prompt_qa.format(
                example = example_qa_by_summary, caption = caps, summary = sum, question = que, option = opt
                ) + "\n" + response_prompt
            response, info = qa_model_by_summary.forward(head=None, prompt=instruction)
            
            response_dict = None
            try: 
                response_dict = ast.literal_eval(response)
            except:
                response_dict = parse_json(response)
            
            if response_dict == None:
                pdb.set_trace()

            with open(f"{output_dir}/{uid}.json", "w") as f:
                json.dump(response_dict, f)

        except openai.RateLimitError:
            print("Too many request: Sleeping 1s", flush=True)
            time.sleep(1)



def summarize_one_video(summarizer, example, video_id, sample_caps, \
                        use_cache, logger):
    caps = ''
    for _, c in sample_caps.items():
        caps += c + "\n"
    instruction = str(video_id) + "\n" + systerm + "\n" + incontext_prompt.format(example = example, caption = caps)
    response, info = summarizer.forward(head = None, prompt = instruction, \
                                        use_cache=use_cache, logger=logger)
    return response

def qa_one_video_by_summary(qa_model, ann, summary, example, video_id, sample_caps, \
                            use_cache, logger):
    
    caps = ''
    for _, c in sample_caps.items():
        caps += c + "\n"

    opt = ''
    que = 'question: ' + ann['question']
    opt += 'option 0: ' + ann['option 0'] + "\n"
    opt += 'option 1: ' + ann['option 1'] + "\n"
    opt += 'option 2: ' + ann['option 2'] + "\n"
    opt += 'option 3: ' + ann['option 3'] + "\n"
    opt += 'option 4: ' + ann['option 4'] + "\n"

    instruction = str(video_id) + "\n" + systerm_qa + "\n" + incontext_prompt_qa.format(
        example = example, caption = caps, summary = summary, question = que, option = opt
        ) + "\n" + response_prompt
    response, info = qa_model.forward(head=None, prompt=instruction, \
                                      use_cache=use_cache, logger=logger)

    response_dict = None
    try: 
        response_dict = ast.literal_eval(response)
    except:
        response_dict = parse_json(response)
    
    if response_dict == None:
        pdb.set_trace()

    return response_dict

def postprocess_response_dict(response_dict):

    if response_dict == None:
        return -1, -1
    
    try:
        id_value = int(response_dict["ANSWER"])
        if id_value < 0 or id_value > 4:
            id_value = -1
        conf = int(response_dict["CONFIDENCE"])
    except:
        pattern = r'\d+'
        match = re.search(pattern, str(response_dict["ANSWER"]))
        conf = re.search(pattern, str(response_dict["CONFIDENCE"]))
        if match:
            id_value = int(match.group())
            conf = int(conf.group())
        else:
            id_value = -1
            conf = -1

    return id_value, conf


if __name__ == "__main__":

    # HCQA_main()

    # exp 1
    interval = 10
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_cap_file = "data/egoschema/lavila_subset.json"
    summary_output_dir = f"results/egoschema/summary/summary_interval={interval}_{timestamp}"  
    qa_output_dir = f"results/egoschema/qa_by_summary/summary_interval={interval}_{timestamp}" 

    # 这一步都是 Cache Hit
    summarize_lavia_subset(summarizer=summarizer, example=example_summary, all_cap_file=all_cap_file, output_dir=summary_output_dir, interval=interval)
    
    # 这一步都是 Cache Miss
    qa_lavia_subset(summary_output_dir, qa_output_dir)

    """
    # exp 2
    # 用 summarize_one_video 和 qa_one_video_by_summary 等价重写 exp1
    init_segments = 19
    all_cap_file = "data/egoschema/lavila_subset.json"
    all_caps = json.load(open(all_cap_file, "r"))
    input_ann_file = "data/egoschema/subset_anno.json"
    all_ann = json.load(open(input_ann_file, "r"))

    summary_num = 0
    summary_frames_all = 0
    total = 0
    no_ans = 0
    correct = 0
    
    for video_id, caps in tqdm(all_caps.items()):
        ann = all_ann[video_id]
        num_frames = len(caps) # 180
        sample_idx = np.linspace(1, num_frames, num=init_segments, dtype=int).tolist()
        sampled_caps = read_caption(caps, sample_idx) 

        summary = summarize_one_video(summarizer, example_summary, video_id, sampled_caps, \
                                    use_cache=True, logger=None) 
        response_dict = qa_one_video_by_summary(summarizer, ann, summary, example_qa_by_summary, video_id, sampled_caps, \
                                                use_cache=True, logger=None)
        answer, confidence = postprocess_response_dict(response_dict)
        
        summary_num += 1  # 现在可以正确修改全局变量
        summary_frames_all += len(sampled_caps)

        label = int(ann['truth'])
        total += 1
        if answer == -1:
            no_ans += 1
        else:
            if answer == label:
                correct += 1

    have_ans = total - no_ans
    acc = correct / have_ans

    print("summary_num", summary_num)
    print("summary_frames_average", summary_frames_all / summary_num)
    print("init_segments", init_segments)

    print("Total: ", total)
    print("No answer: ", no_ans)
    print("have answer: ", total - no_ans)
    print("Mean accuracy (excluded no answer): ", acc)
    print("Mean accuracy (included no answer): ", (have_ans * acc + no_ans * 0.2) / total)
    """

# python3 -m tools.summarizer