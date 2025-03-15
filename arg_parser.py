import argparse

def parse_args():
    parser = argparse.ArgumentParser("TreeVideoAgent")

    # data
    parser.add_argument("--dataset", default='egoschema_subset', type=str)  # 'demo', 'egoschema_subset', 'egoschema', 'nextqa', 'nextgqa', 'intentqa'

    # demo
    parser.add_argument("--demo_info_path", default='data/demo/demo_info.json', type=str)

    # egoschema subset
    parser.add_argument("--data_path", default='data/egoschema/lavila_subset.json', type=str) 
    parser.add_argument("--anno_path", default='data/egoschema/subset_anno.json', type=str)
    parser.add_argument("--duration_path", default='data/egoschema/duration.json', type=str) 

    # 调用 LLM 模型信息
    # gpt-4-1106-preview, gpt-4o-2024-11-20, o3_mini, DeepSeek-R1, llama-3.3-70b
    parser.add_argument("--model_name", default="gpt-4o-2024-11-20", type=str)
    parser.add_argument("--temperature", default=1.0, type=float)   # 2.0
    # cache: cache_gpt4o, cache_o3mini, cache_deepseekr1.pkl, llama_3.3_70b.pkl
    parser.add_argument("--cache_path", default="cache/cache_gpt4o.pkl", type=str)
    parser.add_argument("--use_cache", action='store_false', help="Whether to use llm cache")

    # output / logger
    parser.add_argument("--output_base_path", default="results/egoschema_subset/", type=str)  
    parser.add_argument("--logger_path", default="results/egoschema_subset/", type=str)

    # 迭代与总结设置
    parser.add_argument("--final_step", default=6, type=int)  
    parser.add_argument("--init_interval", default=10, type=int, help='measured by seconds')  # 7, 10
    
    # agent ensemble during serach process 
    parser.add_argument("--s_conf_lower", default=3, type=int, help=">=") # 1,2,3,4,5
    parser.add_argument("--r_conf_lower", default=3, type=int, help=">=") # 1,2,3
    parser.add_argument(
        "--ans_mode", 
        default="vote_conf_and", 
        choices=["s", "r", "sr", "rs", "vote_conf_and", "vote_conf_or"],
        type=str,
        help="s=summarize_and_qa, r=qa_and_reflect")
    
    # post process
    parser.add_argument("--post_s_conf_lower", default=1, type=int, help=">=") # 1,2,3,4,5
    parser.add_argument("--post_r_conf_lower", default=2, type=int, help=">=") # 1,2,3
    parser.add_argument(
        "--post_ans_mode", 
        default="s", 
        choices=["s", "r", "sr", "rs", "vote", "vote_conf_and", "vote_conf_or"],
        type=str,
        help="s=summarize_and_qa, r=qa_and_reflect")
    parser.add_argument("--post_resume_samples", action='store_false', help="Whether to use the init sample_idx in post process")

    # search process
    parser.add_argument(
        "--search_strategy",
        default="a_star",
        choices=["bfs", "gbfs", "dijkstra", "a_star"],
        type=str,
        help="Search strategy to use"
    )
    parser.add_argument("--beam_size", default=5, type=int)  # 在 bfs 中不使用 # [1,5,10]
    parser.add_argument(
        "--for_seg_not_interested",
        default="retain",
        choices=["prune", "retain", "merge"],
        type=str,
        help="for tree node, i.e., video segment not interested, which strategy to use"
    )
    


    # 并行设置
    parser.add_argument("--max_workers", default=3, type=int, help="Number of parallel workers")

    # 特殊例子
    parser.add_argument("--process_num", default=500, type=int)
    parser.add_argument("--specific_id", default=None, type=str)
    parser.add_argument("--specific_id_path", default=None, type=str)  # list 文件的路径
    parser.add_argument("--avoid_id", default=None, type=str)
    parser.add_argument("--reprocess_log", default=None, type=str)

    return parser.parse_args()
