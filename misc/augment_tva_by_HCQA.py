import os, json
import re
import argparse
import pdb
from util import load_json

def build_id_dict(folder_path):
    """
    From eval4_HCQA.py: 
    构建从视频ID到预测结果的字典
    """
    id_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            data = json.load(open(file_path))

            if data == None:
                continue

            try:
                id_value = int(data["ANSWER"])
                if id_value < 0 or id_value > 4:
                    print(filename, data["ANSWER"])
                conf = int(data["CONFIDENCE"])
                id_dict[filename.split('.')[0]] = [id_value, conf]
            except:
                pattern = r'\d+'
                match = re.search(pattern, str(data["ANSWER"]))
                conf = re.search(pattern, str(data["CONFIDENCE"]))
                if match:
                    num = int(match.group())
                    id_dict[filename.split('.')[0]] = [num, conf.group()]

    return id_dict

def main(tva_filepath, hcqa_folder):
    """
    New code:
    结合 TVA 和 HCQA 的结果进行评估
    """
    # 读取 TVA 结果
    tva_data = json.load(open(tva_filepath))
    
    # 读取 HCQA 结果
    hcqa_dict = build_id_dict(hcqa_folder)
    
    # From eval2.py: 基础统计变量
    no_ans = []
    accs = []
    frames = []
    
    # From eval2.py: 统计替换前的结果
    print("\nBefore replacement:")
    for key in tva_data:
        if tva_data[key]["answer"] == -1:
            no_ans.append(key)
            continue
        else:
            acc = tva_data[key]["answer"] == tva_data[key]["label"]
            accs.append(acc)

        frame = tva_data[key]["count_frame"]
        frames.append(frame)

    print("Total: ", len(tva_data))
    print("No answer: ", len(no_ans))
    print("Have answer: ", len(accs))
    print("Mean accuracy (excluded no answer): {:.2f}%".format(sum(accs) / len(accs) * 100))
    print("Mean accuracy (included no answer): {:.2f}%".format((sum(accs) + len(no_ans) * 0.2) / (len(accs) + len(no_ans)) * 100))
    print("Mean frame: {:.2f}".format(sum(frames) / len(frames)))

    # New code: 使用 HCQA 的结果替换 no_ans
    print("\nAfter ensemble:")
    replaced_accs = accs.copy()  # 复制原来的正确结果
    replaced_only_accs = []      # 只包含替换部分的正确率
    replaced_count = 0           # 成功替换的数量
    
    for video_id in no_ans:
        # pdb.set_trace()
        if video_id in hcqa_dict:
            
            replaced_count += 1
            hcqa_answer = hcqa_dict[video_id][0]
            acc = hcqa_answer == tva_data[video_id]["label"]
            replaced_accs.append(acc)
            replaced_only_accs.append(acc)

    print("Total: ", len(tva_data))
    print("No answer replaced: ", replaced_count)
    print("No answer remaining: ", len(no_ans) - replaced_count)
    print("Original accuracy: {:.2f}%".format(sum(accs) / len(accs) * 100))
    if replaced_only_accs:
        print("Replaced part accuracy: {:.2f}%".format(sum(replaced_only_accs) / len(replaced_only_accs) * 100))
    print("Overall accuracy: {:.2f}%".format(
        (sum(replaced_accs) + (len(no_ans) - replaced_count) * 0.2) / len(tva_data) * 100
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble evaluation of TVA and HCQA results")
    parser.add_argument("--tva_path", type=str, required=True,
                      help="Path to the TVA result json file")
    parser.add_argument("--hcqa_folder", type=str, default="results/egoschema/HCQA/summary_interval=10_20250221_231251",
                      help="Path to the folder containing HCQA result files")
    args = parser.parse_args()
    main(args.tva_path, args.hcqa_folder) 