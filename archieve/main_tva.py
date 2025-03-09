import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
import pdb
import numpy as np
from openai import OpenAI
import openai
import pickle
from tqdm import tqdm
from datetime import datetime
from util import *

set_random_seed(42)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger_path = "results/egoschema/ta"
logger = set_logger(timestamp, logger_path)

# TODO remove openai key
# TODO 写一个 argparser
openai.base_url = "https://api.juheai.top/v1/"
openai.api_key = "sk-mHS2kCuqAq2L1uJFE2448bDe4bCc48F8Ab880415D25a7159"
client = OpenAI(api_key=openai.api_key, base_url=openai.base_url)

# 这个函数相当于 VideoTree 中的 model
def get_llm_response(
    system_prompt, prompt, json_format=True, model="gpt-4-1106-preview", use_cache=True
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    key = json.dumps([model, messages])
    logger.info(messages)
    if use_cache:
        cached_value = get_from_cache(key, logger)
        if cached_value is not None:
            logger.info("Cache Hit")
            logger.info(cached_value)
            return cached_value

    logger.info(f"Not hit cache: {key}")

    for _ in range(3):
        try:
            if json_format:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model, messages=messages
                )
            response = completion.choices[0].message.content
            logger.info(response)

            # if use_cache:
            save_to_cache(key, response, logger)
            return response
        except Exception as e:
            logger.error(f"GPT Error: {e}")
            # pdb.set_trace()
            continue
    return "GPT Error"

# LLM Frame Selector
def generate_description_step(question, caption, num_frames, segment_des, use_cache=True):
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "2", "duration": "xxx - xxx", "description": "frame of xxx"},
            {"segment_id": "3", "duration": "xxx - xxx", "description": "frame of xxx"},
        ]
    }
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial frames is not suffient.
    Objective:
    Our goal is to identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Divide the video into segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which segments are likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple frames from one segment if necessary to gather comprehensive insights. 
    Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True, use_cache=use_cache)
    return response


def self_eval(previous_prompt, answer):
    confidence_format = {"confidence": "xxx"}
    prompt = f"""Please assess the confidence level in the decision-making process.
    The provided information is as as follows,
    {previous_prompt}
    The decision making process is as follows,
    {answer}
    Criteria for Evaluation:
    Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
    Partial Information (Confidence Level: 2): If information partially supports an informed guess.
    Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
    Assessment Focus:
    Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the decision-making context.
    Please generate the confidence with JSON format {confidence_format}
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True)
    return response


def generate_answer_cot(question, caption, num_frames, use_cache=True):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think step-by-step and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question.
    """
    system_prompt = "You are a helpful assistant."
    response = get_llm_response(system_prompt, prompt, json_format=False, use_cache=use_cache)
    return prompt, response


def generate_final_answer(question, caption, num_frames, use_cache=True):
    answer_format = {"final_answer": "xxx"}
    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of the sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    Please answer the following question: 
    ``` 
    {question}
    ``` 
    Please think carefully and write the best answer index in Json format {answer_format}. Note that only one answer is returned for the question, and you must select one answer index from the candidates.
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt, json_format=True, use_cache=use_cache)
    return response


def frame_seg_binary(descriptions, video_id, sample_idx):
    # descriptions: 
    # [{'segment_id': '1', 'duration': '1-45', 'description': 'frame of C looking attentively at the laptop, possibly typing or reading content which shows engagement with the task'}, 
    #  {'segment_id': '2', 'duration': '45-90', 'description': 'frame of C interacting with different applications or possibly showing signs of emotion which may indicate increasing engagement or frustration'}, 
    #  {'segment_id': '3', 'duration': '90-135', 'display_description': 'frame of C changing posture or taking a break, which could show a shift in engagement or the need for a pause in the task'}, 
    # pprint(descriptions)
    # sample_idx: [1, 45, 90, 135, 180]
    
    # descriptions 各段中均匀采样
    frame_idx = []
    for idx, description in enumerate(descriptions):
        segment_id = get_segment_id(description)
        seg = segment_id - 1
        seg_frame_idx = (sample_idx[seg] + sample_idx[seg + 1]) // 2
        frame_idx.append(seg_frame_idx)

    return frame_idx


def run_one_question(video_id, ann, caps, logs):
    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5)]
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    num_frames = len(caps)

    ### Step 1 ###
    # TODO：调试这个超参数
    # sample_idx = np.linspace(1, num_frames, num=5, dtype=int).tolist()  # [1, 45, 90, 135, 180]
    sample_idx = np.linspace(1, num_frames, num=9, dtype=int).tolist()
    sampled_caps = read_caption(caps, sample_idx)                       # {'frame 1': '#C C pours the water from the bowl', 'frame 45': '#C C puts the sponge in the sink', 'frame 90': '#C C scrubs the plate with the sponge', 'frame 135': '#C C puts the soap bottle on the sink', 'frame 180': '#C C opens the soap bottle'}
    previous_prompt, answer_str = generate_answer_cot(                  # previous_prompt: "\n    Given a video that has 180 frames, the frames are decoded at 1 fps. Given the following descriptions of five uniformly sampled frames in the video:\n    {'frame 1': '#C C pours the water from the bowl', 'frame 45': '#C C puts the sponge in the sink', 'frame 90': '#C C scrubs the plate with the sponge', 'frame 135': '#C C puts the soap bottle on the sink', 'frame 180': '#C C opens the soap bottle'}\n    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).\n    #O to denote that the sentence is an action done by someone other than the camera wearer.\n    Please answer the following question: \n    ``` \n    Here is the question: Taking into account all the actions performed by c, what can you deduce about the primary objective and focus within the video content?\nHere are the choices: 0. C is cooking. 1. C is doing laundry. 2. C is cleaning the kitchen. 3. C is cleaning dishes. 4. C is cleaning the bathroom.\n    ``` \n    Please think step-by-step and write the best answer index in Json format {'final_answer': 'xxx'}. Note that only one answer is returned for the question.\n    "
        formatted_question, sampled_caps, num_frames                    # 这里已经过了一遍 LLM
    )
    answer = parse_text_find_number(answer_str, logger)                 # answer: 3

    # 自己评估自信度
    confidence_str = self_eval(previous_prompt, answer_str)
    confidence = parse_text_find_confidence(confidence_str, logger)     # confidence: 3

    # TODO 存在超参数：迭代次数
    # until every frame: 7 次迭代
    ### Step 2 ###
    # for step in range(2, 8):
    # TODO 两个超参数单独写一下
    for step in range(2, 7):
        if (step < 5 and confidence < 3) or (answer == -1):
            # print(f"Step:{step} Sample_idx:{sample_idx}")
            try:
                segment_des = {                                         # segment_des: {1: '1-12', 2: '12-23', 3: '23-34', 4: '34-45', 5: '45-56', 6: '56-68', 7: '68-79', 8: '79-90', 9: '90-101', 10: '101-112', 11: '112-123', 12: '123-135', 13: '135-146', 14: '146-157', 15: '157-168', 16: '168-180'}
                    i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                    for i in range(len(sample_idx) - 1)
                }

                # LLM 决定哪些片段需要用, 哪些不需要用
                candidate_descriptions = generate_description_step(
                    formatted_question,
                    sampled_caps,
                    num_frames,
                    segment_des,
                )
                parsed_candidate_descriptions = parse_json(candidate_descriptions)

                # MQA
                # 如果 parse_json 解析不出来，也就是返回的结果是 None，那么就再进行一次生成
                if parsed_candidate_descriptions == None:
                    candidate_descriptions = generate_description_step(
                        formatted_question,
                        sampled_caps,
                        num_frames,
                        segment_des,
                        False,  # use_cache
                    )
                    parsed_candidate_descriptions = parse_json(candidate_descriptions)

                descriptions = get_frames_descriptions(parsed_candidate_descriptions)

                frame_idx = frame_seg_binary(
                    descriptions, video_id, sample_idx
                )
                logger.info(f"Step {step}: {frame_idx}")
                sample_idx += frame_idx
                sample_idx = sorted(list(set(sample_idx)))  # set 已经去重

                sampled_caps = read_caption(caps, sample_idx)
                previous_prompt, answer_str = generate_answer_cot(
                    formatted_question, sampled_caps, num_frames
                )
                answer = parse_text_find_number(answer_str, logger)

                # MQA
                # 如果答案解析不出来，再来一次（Position 1）
                # if answer == -1:
                #     previous_prompt, answer_str = generate_answer_cot(
                #         formatted_question, sampled_caps, num_frames, False
                #     )
                #     answer = parse_text_find_number(answer_str, logger)

                confidence_str = self_eval(previous_prompt, answer_str)
                confidence = parse_text_find_confidence(confidence_str, logger)

            except Exception as e:
                logger.error(f"Step {step} Error: {e}")
                # pdb.set_trace()   # 基本上不会经过了
                answer_str = generate_final_answer(
                    formatted_question, sampled_caps, num_frames
                )
                answer = parse_text_find_number(answer_str, logger) 

                # MQA
                # 如果答案解析不出来，再来一次（Position 2） 
                # if answer == -1:
                #     answer_str = generate_final_answer(
                #         formatted_question, sampled_caps, num_frames, False
                #     )
                #     answer = parse_text_find_number(answer_str, logger)

    # MQA
    # 如果答案解析不出来，再来一次（Position 3）
    if answer == -1:
        answer_str = generate_final_answer(
            formatted_question, sampled_caps, num_frames, False
        )
        answer = parse_text_find_number(answer_str, logger)

    if answer == -1:
        logger.info("Answer Index Not Found!")
        # answer = random.randint(0, 4)   # 这里需要标记一下具体是哪一个问题错了
        answer = -1                       # 把出现错误的 video_id 记录下来

    logger.info(f"Finished video: {video_id}/{answer}/{ann['truth']}")

    label = int(ann["truth"])
    corr = int(label == answer)
    count_frame = len(sample_idx)

    logs[video_id] = {
        "answer": answer,
        "label": label,
        "corr": corr,
        "count_frame": count_frame,
    }


def main():
    # if running full set, change subset to fullset
    input_ann_file = "data/egoschema/subset_anno.json"
    all_cap_file = "data/egoschema/lavila_subset.json"
    json_file_name = f"results/egoschema/ta/ta_subset_{timestamp}.json"  # 写入的结果文件

    anns = json.load(open(input_ann_file, "r"))
    all_caps = json.load(open(all_cap_file, "r"))
    logs = {}

    # 使用 subsub
    # subsub_video_ids = get_video_filenames("data/egoschema/subsub_videos")
    # process_video_ids = get_intersection(subsub_video_ids, list(anns.keys()))

     # 使用 subset
    process_video_ids = list(anns.keys())

    logger.info(f"{len(process_video_ids)} videos to process")
                                                                                  
    tasks = [
        (video_id, anns[video_id], all_caps[video_id], logs)
        for video_id in process_video_ids
    ]

    # 并发执行任务
    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     executor.map(lambda p: run_one_question(*p), tasks)

    # 使用 ThreadPoolExecutor 执行任务并显示进度条
    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     # 使用 tqdm 包装 tasks 以显示进度
    #     for _ in tqdm(executor.map(lambda p: run_one_question(*p), tasks), total=len(tasks), desc="Processing"):
    #         pass

    # 顺序执行任务
    for task in tqdm(tasks):
        run_one_question(*task)
        # pdb.set_trace()

    json.dump(logs, open(json_file_name, "w"))


if __name__ == "__main__":
    main()
