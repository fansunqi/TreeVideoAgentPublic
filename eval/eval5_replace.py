import json
import argparse

def main(file_a, file_b):
    # 读取文件 a 和 b
    data_a = json.load(open(file_a))
    data_b = json.load(open(file_b))

    no_ans = []
    accs = []
    frames = []

    # 用 b 中的预测答案替代 a 中共有项的预测答案
    for key in data_a:
        if key in data_b:
            data_a[key]["answer"] = data_b[key]["answer"]

    # 计算指标
    for key in data_a:
        if data_a[key]["answer"] == -1:
            no_ans.append(key)
            continue
        else:
            acc = data_a[key]["answer"] == data_a[key]["label"]
            accs.append(acc)

        frame = data_a[key]["count_frame"]
        frames.append(frame)

    print("Total: ", len(data_a))
    print("No answer: ", len(no_ans))
    print("Have answer: ", len(accs))
    print("Mean accuracy (included no answer): {:.2f}%".format((sum(accs) + len(no_ans) * 0.2) / (len(accs) + len(no_ans)) * 100))
    print("Mean accuracy (excluded no answer): {:.2f}%".format(sum(accs) / len(accs) * 100))
    print("Mean frame: {:.2f}".format(sum(frames) / len(frames)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace answers from file b into file a and calculate metrics.")
    parser.add_argument("--file_a", type=str, help="Path to the first JSON file (a).")
    parser.add_argument("--file_b", type=str, help="Path to the second JSON file (b).")
    args = parser.parse_args()
    main(args.file_a, args.file_b)