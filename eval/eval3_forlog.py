import os
import re
import argparse

def cal_acc(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    # print(len(lines))
    total = 0
    no_ans = 0
    correct = 0
    # pattern = re.compile(r"- util - INFO - Finished video: (.+)/(-?\d+)/(\d+)")
    pattern = r"Finished video: ([^/]+)/(-?\d+)/(\d+)"

    for line in lines:
        # match = pattern.search(line)
        # 使用正则表达式匹配    
        match = re.search(pattern, line)
        if match:
            # print("HIT: ", line)
            total += 1
            video_id, pred, label = match.groups()
            pred = int(pred)  # 非常重要
            label = int(label)
            if pred == -1:
                no_ans += 1
            if pred == label:
                correct += 1
    have_ans = total - no_ans
    acc_include = correct / total if total > 0 else 0
    acc_exclude = correct / have_ans if have_ans > 0 else 0
    return total, no_ans, have_ans, acc_include, acc_exclude

    

if __name__ == "__main__":
    # 测试正则
    # pattern = re.compile(r"- util - INFO - Finished video: (.+)/(-?\d+)/(\d+)")
    # pattern = r"Finished video: ([^/]+)/(-?\d+)/(\d+)"
    # # 示例日志行
    # log_line = "2025-02-21 20:45:40,882 - util - INFO - Finished video: 03657401-d4a4-40d0-9b03-d7e093ef93d1/-1/0 (line 373)"
    # match = re.search(pattern, log_line)
    # video_id, pred, label = match.groups()
    # pred = int(pred)
    # print(video_id)
    # print(pred)
    # print(pred==-1)
    # print(label)



    # parser = argparse.ArgumentParser(description="Process timestamp for JSON file.")
    # parser.add_argument("date", type=str, help="date(e.g.0220) string for the JSON file")
    # parser.add_argument("timestamp", type=str, help="Timestamp string for the JSON file")
    # args = parser.parse_args()

    # filepath = f"results/egoschema/ta/ta_subset_2025{args.date}_{args.timestamp}.log"

    folder_path = "results/egoschema/ta/"
    files = os.listdir(folder_path)
    last_file = sorted(files)[-1]  # 按名称排序取最后一个
    file_path = os.path.join(folder_path, last_file)
    print(file_path)
    total, no_ans, have_ans, acc_include, acc_exclude = cal_acc(file_path)

    print("Total: ", total)
    print("No answer: ", no_ans)
    print("have answer: ", have_ans)
    print("Mean accuracy (excluded no answer): ", acc_exclude)
    print("Mean accuracy (included no answer): ", acc_include)