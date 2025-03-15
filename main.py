import os
import json
import pdb
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from util import *
from model import GPT
from arg_parser import parse_args
from tools.summarizer import summarize_one_video, \
      qa_one_video_by_summary, postprocess_response_dict
# from tools.utils_clip import get_embeddings, frame_retrieval_seg_ego
from video_seg import *
from arg_parser import parse_args

global_args = parse_args()

# 全局变量 logger 与 timestamp
set_random_seed(42)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger_path = global_args.logger_path
logger = set_logger(timestamp, logger_path)

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

# TODO 之后挪到 main 函数中去
summarizer = GPT(api_key=api_key, model_name=global_args.model_name, temperature=global_args.temperature, base_url=base_url)
qa_model = GPT(api_key=api_key, model_name=global_args.model_name, temperature=global_args.temperature, base_url=base_url)
planner = GPT(api_key=api_key, model_name=global_args.model_name, temperature=global_args.temperature, base_url=base_url)
self_evaluator = GPT(api_key=api_key, model_name=global_args.model_name, temperature=global_args.temperature, base_url=base_url)


# TODO: 使用 promptFactory
def bfs_select_segments(question, caption, num_frames, segment_des, use_cache=True):
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
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response


# 启发代价函数 h(n)：衡量与终点的距离
def gbfs_select_one_segment(question, caption, num_frames, segment_des, use_cache=True):
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"}
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
    Our goal is to step-by-setp identify additional frames that contain crucial information necessary for answering the question. These frames should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames.
    To achieve this, we will:
    1. Consider the video segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which single segment is most likely to contain frames that are most relevant to the question. These frames should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For the segment identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a single sentence per frame. If the specifics of the segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Return the description and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response


# TODO: 搞一下 example，in-context learning
# TODO Dij 也可以用 CLIP、视频段长度等现成工具来写移动代价函数
# 移动代价函数 g(n)：衡量与起点的距离
def dijkstra_select_one_segment(question, caption, num_frames, segment_des, use_cache=True):
    # 一段一段分析某一段，值不值得分割。值得分割说明代价函数小，不值得分割说明代价函数大
    # 这里的分析和目标（也就是问题）无关

    # 下面这一段 Prompt 需要好好琢磨一下
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"}
        ]
    }

    prompt = f"""
    Given a video that has {num_frames} frames, the frames are decoded at 1 fps. Given the following descriptions of sampled frames in the video:
    {caption}
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head).
    #O to denote that the sentence is an action done by someone other than the camera wearer.
    
    Based on the intervals between the sample frames, we have candiate video segments: {segment_des}
    
    Please identify which candidate video segment contains richest visual elements and most dramatic scene changes, making it most suitable for splitting into smaller video segments. For example, if the characters and scenes have changed between the two sampled frames, then this video segment is suitable for splitting into smaller and atomic video segments.
    For the segment identified as most suitable for splitting into smaller video segments, provide a concise description focusing on the segment's rich visual elements or scene changes. If the specifics of the segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects.
    Return the description and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response


# 同时考虑两件事情的就是 A* 搜索
# 代价函数 f(n) = 启发代价函数 h(n) + 移动代价函数 g(n)
def a_star_select_one_segment(question, caption, num_frames, segment_des, use_cache=True):
    formatted_description = {
        "frame_descriptions": [
            {"segment_id": "1", "duration": "xxx - xxx", "description": "frame of xxx"}
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
    In order to obtain more information about the video and ultimately answer the question, we need to step-by-setp identify video segment between the initial frames that meet the following two conditions:
    1. contains crucial information necessary for answering the question. This video segment should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial frames. This segment should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    2. contains rich visual elements and dramatic scene changes, making it suitable for splitting into smaller video segments. For example, if the characters and scenes have changed between the two sampled frames, then this video segment is suitable for splitting into smaller and atomic video segments.

    To achieve this, we will:
    1. Consider the video segments based on the intervals between the initial frames as, candiate segments: {segment_des}
    2. Determine which single candidate segment is most likely to meet the above two conditions.
    For the segment identified as most suitable, provide a concise description focusing on essential visual elements in the segment. Use a single sentence per frame. If the specifics of the segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects.
    Return the description and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response



def self_eval(previous_prompt, answer, use_cache=True):
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
    response, _ = self_evaluator.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
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
    response, _ = qa_model.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=False)
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
    response, _ = qa_model.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response


def summarize_and_qa(video_id, sampled_caps, ann, args):
    summary = summarize_one_video(summarizer, video_id, sampled_caps, \
                                        use_cache=args.use_cache, logger=logger)
    response_dict = qa_one_video_by_summary(qa_model, ann, summary, video_id, sampled_caps, \
                                            use_cache=args.use_cache, logger=logger)
    # 后解析 repsonse_dict
    answer, confidnce = postprocess_response_dict(response_dict)
    return answer, confidnce


def qa_and_reflect(formatted_question, sampled_caps, num_frames, args):
    previous_prompt, answer_str = generate_answer_cot(                  
        formatted_question, sampled_caps, num_frames, args.use_cache                   
    )
    answer = parse_text_find_number(answer_str, logger)                

    # 自己评估自信度
    confidence_str = self_eval(previous_prompt, answer_str, args.use_cache)
    confidence = parse_text_find_confidence(confidence_str, logger)

    return answer, confidence


def choose_ans(s_qa_ans, s_qa_conf, s_conf_lower, \
               r_qa_ans, r_qa_conf, r_conf_lower, \
                ans_mode, step):
    
    answer = -1
    get_ans_step = None 

    if ans_mode == "s":
        if s_qa_ans != -1 and s_qa_conf >= s_conf_lower:
            answer = s_qa_ans
            get_ans_step = f"{step}_s_qa"
    
    elif ans_mode == "r":
        if r_qa_ans != -1 and r_qa_conf >= r_conf_lower:
            answer = r_qa_ans
            get_ans_step = f"{step}_r_qa"
    
    elif ans_mode == "sr":
        # 优先 s
        if s_qa_ans != -1 and s_qa_conf >= s_conf_lower:
            answer = s_qa_ans
            get_ans_step = f"{step}_s_qa"
        
        elif r_qa_ans != -1 and r_qa_conf >= r_conf_lower:
            answer = r_qa_ans
            get_ans_step = f"{step}_r_qa"
    
    elif ans_mode == "rs":
        # 优先 r
        if r_qa_ans != -1 and r_qa_conf >= r_conf_lower:
            answer = r_qa_ans
            get_ans_step = f"{step}_r_qa"

        elif s_qa_ans != -1 and s_qa_conf >= s_conf_lower:
            answer = s_qa_ans
            get_ans_step = f"{step}_s_qa"

    # vote 的关键就是必须两个答案一致才返回
    elif ans_mode == "vote":
        if s_qa_ans == r_qa_ans:
            answer = s_qa_ans
            get_ans_step = f"{step}_s_r"
    
    elif (ans_mode == "vote_conf_and"):
        if (s_qa_ans == r_qa_ans) and \
           ((s_qa_conf >= s_conf_lower) and (r_qa_conf >= r_conf_lower)):
            answer = s_qa_ans
            get_ans_step = f"{step}_s_r"
    
    elif (ans_mode == "vote_conf_or"):
        if (s_qa_ans == r_qa_ans) and \
           ((s_qa_conf >= s_conf_lower) or (r_qa_conf >= r_conf_lower)):
            answer = s_qa_ans
            get_ans_step = f"{step}_s_r"

    else:
        raise KeyError # 说明参数输错
    
    return answer, get_ans_step


# TODO: LLM 直接想象中间哪一帧比较重要，而不是二分插值
# TODO: 用 CLIP 插帧
# TODO: 间隔比较大就多插帧，反之，少插帧
def split_and_reconnect_segments(selected_video_segments, video_segments, for_seg_not_interested, num_frames):

    # TODO 确保 video_segments 一定是有序，无重复的

    new_segments = []

    if for_seg_not_interested == "prune":
    
        # 对每个选中的视频切片进行二分
        for segment in selected_video_segments:
            
            if segment.start >= segment.end - 1:
                # 如果 seg 只有一或两张图片，就不可分了
                new_segments.append(segment)
            else:
                mid_point = (segment.start + segment.end) // 2  # 计算中点

                # 创建两个新的 VideoSeg 实例
                first_half = VideoSeg(start=segment.start, end=mid_point)
                second_half = VideoSeg(start=mid_point, end=segment.end)
                
                # 将新生成的两个切片添加到结果列表中
                new_segments.append(first_half)
                new_segments.append(second_half)
    
    elif for_seg_not_interested == "retain":

        for segment in video_segments:

            if segment in selected_video_segments:
                if segment.start >= segment.end - 1:
                    # 如果 seg 只有一张图片，就不可分了
                    new_segments.append(segment)
                else:
                    mid_point = (segment.start + segment.end) // 2  # 计算中点

                    # 创建两个新的 VideoSeg 实例
                    first_half = VideoSeg(start=segment.start, end=mid_point)
                    second_half = VideoSeg(start=mid_point, end=segment.end)
                    
                    # 将新生成的两个切片添加到结果列表中
                    new_segments.append(first_half)
                    new_segments.append(second_half)
            else:
                new_segments.append(segment)

    elif for_seg_not_interested == "merge":
        
        for i, segment in enumerate(selected_video_segments):

            if i == 0:
                # 把头部那一段连上
                if segment.start != 1:
                    video_start_seg = VideoSeg(start=1, end=segment.start)
                    new_segments.append(video_start_seg)

            # 把之前缺失的若干段 merge 成一个新节点
            if i != 0 and segment.start != new_segments[-1].end:
                video_merged_seg = VideoSeg(start=new_segments[-1].end, end=segment.start)
                new_segments.append(video_merged_seg)


            if segment.start >= segment.end - 1:
                # 如果 seg 只有一张图片，就不可分了
                new_segments.append(segment)
            else:
                mid_point = (segment.start + segment.end) // 2  # 计算中点

                # 创建两个新的 VideoSeg 实例
                first_half = VideoSeg(start=segment.start, end=mid_point)
                second_half = VideoSeg(start=mid_point, end=segment.end)
                
                # 将新生成的两个切片添加到结果列表中
                new_segments.append(first_half)
                new_segments.append(second_half)

            if i == len(selected_video_segments) - 1:
                # 尾部一段也连上
                if segment.start != 180:
                    video_start_seg = VideoSeg(start=segment.end, end=num_frames)
                    new_segments.append(video_start_seg)
            
    else:
        raise KeyError
    
    return new_segments


def frame_seg_clip(video_segments, frame_embeddings):
    new_segments = []

    # frame_embeddings = np.load(f"data/egoschema/ego_features_448/{video_id}.npy")

    for segment in video_segments:
        seg_frame_embeddings = frame_embeddings[segment.start : segment.end]


def select_process(formatted_question, sample_idx, sampled_caps, num_frames, step, 
                   args, all_sample_idx, caps, video_segments, select_fn):
    
    segment_des = {
        i + 1: f"{video_seg.start}-{video_seg.end}"
        for i, video_seg in enumerate(video_segments)
    }
    # # segment_des: {1: '1-12', 2: '12-23', 3: '23-34', 4: '34-45', ...

    # LLM 决定 segment_des 中哪些片段需要用, 哪些不需要用          
    candidate_descriptions = select_fn(formatted_question, sampled_caps, num_frames, segment_des, args.use_cache)
    if candidate_descriptions != None:
        parsed_candidate_descriptions = parse_json(candidate_descriptions)
        selected_descriptions = get_frames_descriptions(parsed_candidate_descriptions)

    # 如果 LLM 没有选出来，就再来一次
    max_generate = 5
    generate_count = 0
    while candidate_descriptions == None or selected_descriptions == None:
        generate_count += 1
        if generate_count > max_generate:
            break
        candidate_descriptions = select_fn(formatted_question, sampled_caps, num_frames, segment_des, False)
        parsed_candidate_descriptions = parse_json(candidate_descriptions)
        selected_descriptions = get_frames_descriptions(parsed_candidate_descriptions)

    # 根据 selected_descriptions 提取 VideoSeg 实例
    selected_video_segments = extract_videoseg_from_descriptions(selected_descriptions)

    video_segments = split_and_reconnect_segments(selected_video_segments, video_segments, args.for_seg_not_interested, num_frames)

    # 从 new_segments 列表中提取采样帧，并去重。
    sample_idx_set = set()
    for segment in video_segments:
        sample_idx_set.add(segment.start)  
        sample_idx_set.add(segment.end)   
    sample_idx = sorted(list(sample_idx_set))  
    
    return video_segments, sample_idx


def run_one_question(video_id, ann, caps, logs, args):

    logger.info(f"Start to process {video_id}")
    print(f"Start video: {video_id}")
    
    get_ans_step = None          # 统计在哪一步得出答案
    sample_idx_change_list = []  # 统计 sample_idx 的变化情况 

    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5)]
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    num_frames = len(caps)

    # Root node 初始化
    sample_idx = list(range(1, num_frames + 1, args.init_interval))     # 从 1 开始，到 num_frames + 1，步长为 interval
    # TODO 可以在这里加上最末尾一帧

    sampled_caps = read_caption(caps, sample_idx)                       # {'frame 1': '#C C pours the water from the bowl', 'frame 45': '#C C puts the sponge in the sink', 'frame 90': '#C C scrubs the plate with the sponge', 'frame 135': '#C C puts the soap bottle on the sink', 'frame 180': '#C C opens the soap bottle'}
    all_sample_idx = sample_idx
    # 初始化 video_segments 列表
    video_segments = []   
    for segment_id in range(1, len(sample_idx)):
        video_seg = VideoSeg(sample_idx[segment_id - 1], sample_idx[segment_id], segment_id, None)
        video_segments.append(video_seg)
    sample_idx_change_list.append(sample_idx)
    

    # 树搜索主循环 1. LLM QA 2. 选节点分裂
    for step in range(1, args.final_step + 1):

        # print(f"{video_id}: step {step}, sample_idx {sample_idx}")
        print_segment_list(video_segments)

        # 1. LLM QA
        if args.ans_mode == "s":
            s_qa_ans, s_qa_conf = summarize_and_qa(video_id, sampled_caps, ann, args)
            r_qa_ans, r_qa_conf = None, None
        elif args.ans_mode == "r":
            r_qa_ans, r_qa_conf = qa_and_reflect(formatted_question, sampled_caps, num_frames, args)
            s_qa_ans, s_qa_conf = None, None
        else:
            s_qa_ans, s_qa_conf = summarize_and_qa(video_id, sampled_caps, ann, args)
            r_qa_ans, r_qa_conf = qa_and_reflect(formatted_question, sampled_caps, num_frames, args)

        answer, get_ans_step = choose_ans(s_qa_ans, s_qa_conf, args.s_conf_lower, \
                                          r_qa_ans, r_qa_conf, args.r_conf_lower, \
                                          args.ans_mode, step)

        if answer != -1:
            break   # 循环结束
        
        # 2. 选节点分裂
        if args.search_strategy == "bfs": 
            select_fn = bfs_select_segments
            video_segments, sample_idx = \
                select_process(formatted_question, sample_idx, sampled_caps, num_frames, 
                               step, args, all_sample_idx, caps, video_segments, select_fn)
        
        elif args.search_strategy == "gbfs": 
            select_fn = gbfs_select_one_segment
            for select_iter in range(args.beam_size):
                video_segments, sample_idx = \
                    select_process(formatted_question, sample_idx, sampled_caps, num_frames, 
                                   step, args, all_sample_idx, caps, video_segments, select_fn)
        elif args.search_strategy == "dijkstra":
            select_fn = dijkstra_select_one_segment
            for select_iter in range(args.beam_size):
                video_segments, sample_idx = \
                    select_process(formatted_question, sample_idx, sampled_caps, num_frames, 
                                   step, args, all_sample_idx, caps, video_segments, select_fn)
        elif args.search_strategy == "a_star":
            select_fn = a_star_select_one_segment
            for select_iter in range(args.beam_size):
                video_segments, sample_idx = \
                    select_process(formatted_question, sample_idx, sampled_caps, num_frames, 
                                   step, args, all_sample_idx, caps, video_segments, select_fn)
        else:
            raise KeyError
            
        # 合并到 all_sample_idx
        all_sample_idx = sorted(list(set(all_sample_idx + sample_idx)))
        sample_idx_change_list.append(sample_idx)
        sampled_caps = read_caption(caps, sample_idx)

    # 树搜索完成，进入收尾
    # print(video_id, "final sample num:", len(sample_idx))

    # Post Process 阶段
    if answer == -1:
        if args.post_resume_samples:
            sample_idx = list(range(1, num_frames + 1, args.init_interval))     # 从 1 开始，到 num_frames + 1，步长为 interval
            sample_idx_change_list.append(sample_idx)
            sampled_caps = read_caption(caps, sample_idx)
        else:
            sample_idx = all_sample_idx  


        if args.post_ans_mode == "s":
            s_qa_ans, s_qa_conf = summarize_and_qa(video_id, sampled_caps, ann, args)
            r_qa_ans, r_qa_conf = None, None
        elif args.post_ans_mode == "r":
            r_qa_ans, r_qa_conf = qa_and_reflect(formatted_question, sampled_caps, num_frames, args)
            s_qa_ans, s_qa_conf = None, None
        else:
            s_qa_ans, s_qa_conf = summarize_and_qa(video_id, sampled_caps, ann, args)
            r_qa_ans, r_qa_conf = qa_and_reflect(formatted_question, sampled_caps, num_frames, args)
        step = "post"
        answer, get_ans_step = choose_ans(s_qa_ans, s_qa_conf, args.post_s_conf_lower, \
                                          r_qa_ans, r_qa_conf, args.post_r_conf_lower, \
                                          args.post_ans_mode, step)

    # final QA，直接强出答案
    if answer == -1:
        # print("final_direct_qa")
        answer_str = generate_final_answer(
            formatted_question, sampled_caps, num_frames, args.use_cache
        )
        answer = parse_text_find_number(answer_str, logger)
        get_ans_step = f"final_direct_qa"

    # no_ans 处理
    if answer == -1:
        logger.info("Answer Index Not Found!")
        # answer = random.randint(0, 4)       # 这里需要标记一下具体是哪一个问题错了
        answer = -1                       
        print(f"No ans video id: {video_id}") # 把出现错误的 video_id 打印出来

    logger.info(f"Finished video: {video_id}/{answer}/{ann['truth']}")
    print(f"Finished video: {video_id}/{answer}/{ann['truth']}")
    # print_nested_list(sample_idx_change_list)

    label = int(ann["truth"])
    corr = int(label == answer)
    count_frame = len(all_sample_idx)             # 这里计算总共使用了多少帧

    logs[video_id] = {
        "answer": answer,
        "label": label,
        "corr": corr,
        "count_frame": count_frame,
        "get_ans_step": get_ans_step,
    }


def main(args):

    output_result_file = os.path.join(args.output_base_path, f"{timestamp}.json")

    anns = json.load(open(args.anno_path, "r"))
    all_caps = json.load(open(args.data_path, "r"))
    logs = {}

    process_video_ids = list(anns.keys())[:args.process_num]

    # 特殊处理一个视频
    if args.specific_id != None:  
        specific_video_ids = [args.specific_id]
        process_video_ids = get_intersection(specific_video_ids, list(anns.keys()))
    
    if args.reprocess_log != None:
        # Load the log file and find video_ids with answer == -1
        reprocess_log = json.load(open(args.reprocess_log, "r"))
        process_video_ids = [
            video_id for video_id, log in reprocess_log.items() 
            # if log["answer"] == -1
            if log["answer"] != log["label"]
        ]
    
    if args.avoid_id != None:
        process_video_ids.remove(args.avoid_id)
    
    if args.specific_id_path != None:
        process_video_ids = json.load(open(args.specific_id_path, "r"))


    logger.info(f"{len(process_video_ids)} videos to process")
                                                                                  
    tasks = [
        (video_id, anns[video_id], all_caps[video_id], logs, args)
        for video_id in process_video_ids
    ]

    # 并发执行任务, 并显示进度条
    # with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    #     # 使用 tqdm 包装 tasks 以显示进度
    #     for _ in tqdm(executor.map(lambda p: run_one_question(*p), tasks), total=len(tasks), desc="Processing"):
    #         pass
    
    for task in tqdm(tasks):
        try:
            run_one_question(*task)
        except Exception as e:

            # TODO 乱蒙一个
            
            print(f"\nError -- main -- {e}\n")

    json.dump(logs, open(output_result_file, "w"))


def demo(args):

    output_result_file = os.path.join(args.output_base_path, f"{timestamp}.json")

    demo_info = json.load(open(args.demo_info_path, "r"))
    anns = demo_info["anns"]
    all_caps = demo_info["all_caps"]
    
    logs = {}
    logger.info("process demo video...")
                                                                                  
    run_one_question("demo", anns, all_caps, logs, args)
    json.dump(logs, open(output_result_file, "w"))


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    if args.dataset == 'demo':
        demo(args)

    else:
        main(args)
        
        print(args)
        print(f"\ntimestamp: {timestamp}\n")

        # eval
        os.system(f"python3 eval.py results/{args.dataset}/{timestamp}.json")

        # visualize
        # os.system(f"python3 visualize/get_ans_step.py --filename ta_subset_{timestamp}.json")