import os
import json
import pdb
import numpy as np
from openai import OpenAI
import openai
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
import matplotlib
import matplotlib.pyplot as plt
import seaborn


# 全局变量 logger 与 timestamp
set_random_seed(42)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger_path = "results/egoschema/ta"  # TODO logger 写入 argparser
logger = set_logger(timestamp, logger_path)

# TODO 看一下 openai 会员
# TODO 之后从环境变量中找 api_keys
api_key = "sk-LXxWmvnd5yYm79eg8d2b5d8350Ca400aA4C7BaAd072aE63c"
base_url = "https://api.juheai.top/v1/"
summarizer = GPT(api_key=api_key, model_name="gpt-4-1106-preview", base_url=base_url)
qa_model = GPT(api_key=api_key, model_name="gpt-4-1106-preview", base_url=base_url)

# TODO 把 planner 改成是 qa_model + evaluator
planner = GPT(api_key=api_key, model_name="gpt-4-1106-preview", base_url=base_url)


# TODO: 使用 promptFactory
def llm_select_segment(question, caption, num_frames, segment_des, use_cache=True):
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

# TODO: 每次只选最相关的一段，然后再选一段，再选一段, ..., 逐渐密布
# 启发式函数 f(n)
def llm_select_one_segment(question, caption, num_frames, segment_des, use_cache=True):
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


# TODO: 单独评价一段视频切片的相关性，然后再排序选切片
# def llm_evaluate_segment():
#     pass


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
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
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
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=False)
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
    response, _ = planner.forward(head=system_prompt, prompt=prompt, logger=logger, use_cache=use_cache, use_json_format=True)
    return response

# TODO: LLM 直接想象中间哪一帧比较重要，而不是二分插值
# TODO: 用 CLIP 插帧
# TODO: 间隔比较大就多插帧，反之，少插帧
def frame_seg_binary(video_segments):
    """
    对每个 VideoSeg 实例进行二分操作，将其拆分成两个更小的 VideoSeg 实例。
    
    :param video_segments: List[VideoSeg], 包含多个 VideoSeg 实例的列表。
    :return: List[VideoSeg], 返回拆分后的新的 VideoSeg 实例列表。
    """
    new_segments = []
    
    # 对每个视频切片进行二分
    for segment in video_segments:
        mid_point = (segment.start + segment.end) // 2  # 计算中点
        
        # 创建两个新的 VideoSeg 实例
        first_half = VideoSeg(start=segment.start, end=mid_point)
        second_half = VideoSeg(start=mid_point + 1, end=segment.end)
        
        # 将新生成的两个切片添加到结果列表中
        new_segments.append(first_half)
        new_segments.append(second_half)
    
    return new_segments

def frame_seg_clip(video_segments, frame_embeddings):
    new_segments = []

    # frame_embeddings = np.load(f"data/egoschema/ego_features_448/{video_id}.npy")

    for segment in video_segments:
        seg_frame_embeddings = frame_embeddings[segment.start : segment.end]



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


def select_process(formatted_question, sampled_caps, num_frames, segment_des, 
                   args, all_sample_idx, caps, select_fn):
    # LLM 决定 segment_des 中哪些片段需要用, 哪些不需要用          
    candidate_descriptions = select_fn(formatted_question, sampled_caps, num_frames, segment_des, args.use_cache)
    parsed_candidate_descriptions = parse_json(candidate_descriptions)
    selected_descriptions = get_frames_descriptions(parsed_candidate_descriptions)

    # re-select: 如果结果是 None，那么就不使用 cache 再进行一次挑选
    if selected_descriptions == None:
        candidate_descriptions = select_fn(formatted_question, sampled_caps, num_frames, segment_des, False)
        parsed_candidate_descriptions = parse_json(candidate_descriptions)
        selected_descriptions = get_frames_descriptions(parsed_candidate_descriptions)

    
    # 根据 selected_descriptions 提取 VideoSeg 实例
    video_segments = extract_videoseg_from_descriptions(selected_descriptions)

    # 调用 frame_seg_binary 对每个视频切片进行二分
    video_segments = frame_seg_binary(video_segments)
    
    # 从 new_segments 列表中提取采样帧，并去重。
    sample_idx_set = set()
    for segment in video_segments:
        sample_idx_set.add(segment.start)  
        sample_idx_set.add(segment.end)   
    sample_idx = sorted(list(sample_idx_set))

    # 合并到 all_sample_idx
    all_sample_idx = sorted(list(set(all_sample_idx + sample_idx)))
    sampled_caps = read_caption(caps, sample_idx)
    
    '''
    frame_idx = frame_retrieval_seg_ego(
            selected_descriptions, video_id, sample_idx
        )
    sample_idx += frame_idx
    sample_idx = sorted(list(set(sample_idx)))
    all_sample_idx = sorted(list(set(all_sample_idx + sample_idx)))
    sampled_caps = read_caption(caps, sample_idx)
    '''
    return video_segments, sample_idx, all_sample_idx, sampled_caps


def run_one_question(video_id, ann, caps, logs, args):

    logger.info(f"Start to process {video_id}")
    
    get_ans_step = None  # 统计在哪一步得出答案

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
    sampled_caps = read_caption(caps, sample_idx)                       # {'frame 1': '#C C pours the water from the bowl', 'frame 45': '#C C puts the sponge in the sink', 'frame 90': '#C C scrubs the plate with the sponge', 'frame 135': '#C C puts the soap bottle on the sink', 'frame 180': '#C C opens the soap bottle'}
    all_sample_idx = sample_idx


    # 树搜索主循环 1. LLM QA 2. 分裂节点
    for step in range(1, args.final_step + 1):

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
        
        # 2. 分裂节点
        # TODO：利用 new_segments 重写
        
        if args.retain_seg or step == 1:
            segment_des = {                                         # segment_des: {1: '1-12', 2: '12-23', 3: '23-34', 4: '34-45', 5: '45-56', 6: '56-68', 7: '68-79', 8: '79-90', 9: '90-101', 10: '101-112', 11: '112-123', 12: '123-135', 13: '135-146', 14: '146-157', 15: '157-168', 16: '168-180'}
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }

        else:
            segment_des = {
                i + 1: f"{video_seg.start}-{video_seg.end}"
                for i, video_seg in enumerate(video_segments)
            }
        
        if args.search_strategy == "bfs": 
            select_fn = llm_select_segment
            video_segments, sample_idx, all_sample_idx, sampled_caps = \
                select_process(formatted_question, sampled_caps, num_frames, segment_des, 
                   args, all_sample_idx, caps, select_fn)
        
        elif args.search_strategy == "gbfs": 
            select_fn = llm_select_one_segment
            for select_iter in range(args.select_num_one_step):
                video_segments, sample_idx, all_sample_idx, sampled_caps = \
                    select_process(formatted_question, sampled_caps, num_frames, segment_des, 
                    args, all_sample_idx, caps, select_fn)
        else:
            raise KeyError


    # 树搜索完成，进入收尾
    # print(video_id, "final sample num:", len(sample_idx))

    
    # Post Process 阶段
    # TODO NOTE 使用 all_sample_idx 再过一遍
    # sample_idx = all_sample_idx
    
    # 对比试验
    sample_idx = list(range(1, num_frames + 1, args.init_interval))     # 从 1 开始，到 num_frames + 1，步长为 interval  

    if answer == -1:
        sampled_caps = read_caption(caps, sample_idx)

        if args.post_ans_mode == "s":
            s_qa_ans, s_qa_conf = summarize_and_qa(video_id, sampled_caps, ann, args)
            r_qa_ans, r_qa_conf = None, None
        elif args.post_ans_mode == "r":
            r_qa_ans, r_qa_conf = qa_and_reflect(formatted_question, sampled_caps, num_frames, args)
            s_qa_ans, s_qa_conf = None, None
        else:
            s_qa_ans, s_qa_conf = summarize_and_qa(video_id, sampled_caps, ann, args)
            r_qa_ans, r_qa_conf = qa_and_reflect(formatted_question, sampled_caps, num_frames, args)
        step = "final"
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

    output_result_file = os.path.join(args.output_base_path, f"ta_subset_{timestamp}.json")

    anns = json.load(open(args.anno_path, "r"))
    all_caps = json.load(open(args.data_path, "r"))
    logs = {}

    process_video_ids = list(anns.keys())

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


    logger.info(f"{len(process_video_ids)} videos to process")
                                                                                  
    tasks = [
        (video_id, anns[video_id], all_caps[video_id], logs, args)
        for video_id in process_video_ids
    ]

    # 并发执行任务, 并显示进度条
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 使用 tqdm 包装 tasks 以显示进度
        for _ in tqdm(executor.map(lambda p: run_one_question(*p), tasks), total=len(tasks), desc="Processing"):
            pass

    json.dump(logs, open(output_result_file, "w"))


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    main(args)
    
    print(args)
    print(f"\ntimestamp: {timestamp}\n")

    # eval
    os.system(f"python3 -m eval.eval2 results/egoschema/ta/ta_subset_{timestamp}.json")

    # visualize
    os.system(f"python3 visualize/get_ans_step.py --filename ta_subset_{timestamp}.json")



# TODO: 多个答案投票
# TODO: 比较一下 summary-QA 和 Reflect—QA 到底哪一个更好

# TODO: 用 HCGQ 或者 VideoAgent 的方法跑一下
# TODO 思考一下 online 怎么做

# CLIP 也太慢了，不太行啊