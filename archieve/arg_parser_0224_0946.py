import argparse

def parse_args():
    parser = argparse.ArgumentParser("TreeVideoAgent")

    # data
    parser.add_argument("--dataset", default='egoschema', type=str)  # 'egoschema', 'nextqa', 'nextgqa', 'intentqa'

    # egoschema subset
    parser.add_argument("--data_path", default='data/egoschema/lavila_subset.json', type=str) 
    parser.add_argument("--anno_path", default='data/egoschema/subset_anno.json', type=str)
    parser.add_argument("--duration_path", default='data/egoschema/duration.json', type=str) 

    # TODO 加入调用 LLM 模型名字

    # output
    parser.add_argument("--output_base_path", default="results/egoschema/ta/", type=str)  

    # 迭代与总结设置
    parser.add_argument("--final_step", default=6, type=int)  
    parser.add_argument("--init_interval", default=10, type=int)
    
    parser.add_argument("--s_conf_lower", default=3, type=int, help=">=") # 1,2,3,4,5
    parser.add_argument("--r_conf_lower", default=3, type=int, help=">=") # 1,2,3
    parser.add_argument(
        "--ans_mode", 
        default="vote_conf_and", 
        choices=["s", "r", "sr", "rs", "vote_conf_and", "vote_conf_or"],
        type=str,
        help="s=summarize_and_qa, r=qa_and_reflect")
    
    # post_process
    parser.add_argument("--post_s_conf_lower", default=1, type=int, help=">=") # 1,2,3,4,5
    parser.add_argument("--post_r_conf_lower", default=2, type=int, help=">=") # 1,2,3
    parser.add_argument(
        "--post_ans_mode", 
        default="s", 
        choices=["s", "r", "sr", "rs", "vote", "vote_conf_and", "vote_conf_or"],
        type=str,
        help="s=summarize_and_qa, r=qa_and_reflect")
    
    parser.add_argument("--retain_seg", action='store_true', 
                        help="Whether to retain currently not interested segments")

    parser.add_argument(
        "--search_strategy",
        default="bfs",
        choices=["bfs", "gbfs", "dijkstra"],
        type=str,
        help="Search strategy to use"
    )
    parser.add_argument("--select_num_one_step", default=1, type=int)  # 3
    
    # cache 
    parser.add_argument("--cache_path", default="cache/egoschema/tva_cache_gpt4.pkl", type=str)
    parser.add_argument("--use_cache", action='store_false', help="Whether to use llm cache")

    # 并行设置
    parser.add_argument("--max_workers", default=5, type=int, help="Number of parallel workers")

    # 特殊例子
    parser.add_argument("--process_num", default=300, type=int)
    parser.add_argument("--specific_id", default=None, type=str)
    parser.add_argument("--reprocess_log", default=None, type=str)

    return parser.parse_args()
