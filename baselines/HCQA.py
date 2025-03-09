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
from util import parse_json


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


api_key = "sk-77M5GBxTX9WS4JBZ693c6bF358F64d369dF8778e2070Db31"
base_url = "https://api.juheai.top/v1/"
model_name = "gpt-4-1106-preview"
summarizer = GPT(api_key=api_key, model_name=model_name, base_url=base_url)
qa_model_by_summary = GPT(api_key=api_key, model_name=model_name, base_url=base_url)



def summarize_lavia_subset(all_cap_file, output_dir, interval):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_caps = json.load(open(all_cap_file, "r"))
    for video_id, captions in tqdm(all_caps.items()):
        num_frames = len(captions) # 180
        caps = ''
        for c_id, c in enumerate(captions):
            if c_id % interval == 0:
                caps += c + "\n"
        instruction = str(video_id) + "\n" + systerm + "\n" + incontext_prompt.format(example=example_summary, caption=caps)
        response, info = summarizer.forward(head=None, prompt=instruction)
        with open(f"{output_dir}/{video_id}.txt", "w") as f:
            f.write(response)


def qa_lavia_subset(summary_path, output_dir, interval):
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
            # 一开始区别是在于这里用了所有 captions
            # 而不是截取的 captions
            # 现已修正
            captions = all_caps[uid]
            caps = ''
            for c_id, c in enumerate(captions):
                if c_id % interval == 0:
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


if __name__ == "__main__":

    interval = 10

    # 读取 TreeVideoAgent 的某一个 json 结果文件中的 no_ans
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_cap_file = "data/egoschema/lavila_subset.json"
    summary_output_dir = f"results/egoschema/summary/summary_interval={interval}_{timestamp}"  
    qa_output_dir = f"results/egoschema/qa_by_summary/summary_interval={interval}_{timestamp}" 

    summarize_lavia_subset(all_cap_file=all_cap_file, output_dir=summary_output_dir, interval=interval)

    qa_lavia_subset(summary_output_dir, qa_output_dir, interval)

