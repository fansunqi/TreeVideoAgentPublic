import json
import argparse

def main(filepath, va_stf_path):
    data = json.load(open(filepath))
    va_stf_data = json.load(open(va_stf_path))

    no_ans = []
    accs = []
    frames = []
    
    # 统计替换前的结果
    print("\nBefore replacement:")
    for key in data:
        if data[key]["answer"] == -1:
            no_ans.append(key)
            continue
        else:
            acc = data[key]["answer"] == data[key]["label"]
            accs.append(acc)

        frame = data[key]["count_frame"]
        frames.append(frame)

    print("Total: ", len(data))
    print("No answer: ", len(no_ans))
    print("Have answer: ", len(accs))
    print("Mean accuracy (excluded no answer): {:.2f}%".format(sum(accs) / len(accs) * 100))
    print("Mean accuracy (included no answer): {:.2f}%".format((sum(accs) + len(no_ans) * 0.2) / (len(accs) + len(no_ans)) * 100))
    print("Mean frame: {:.2f}".format(sum(frames) / len(frames)))

    # 使用 va_stf 的结果替换 no_ans
    print("\nAfter replacement:")
    replaced_accs = accs.copy()  # 复制原来的正确结果
    replaced_only_accs = []      # 只包含替换部分的正确率
    
    for video_id in no_ans:
        acc = va_stf_data[video_id]["answer"] == va_stf_data[video_id]["label"]
        replaced_accs.append(acc)
        replaced_only_accs.append(acc)

    print("Total: ", len(data))
    print("Original accuracy: {:.2f}%".format(sum(accs) / len(accs) * 100))
    print("Replaced part accuracy: {:.2f}%".format(sum(replaced_only_accs) / len(replaced_only_accs) * 100))
    print("Overall accuracy: {:.2f}%".format(sum(replaced_accs) / len(replaced_accs) * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", default="", type=str)
    parser.add_argument("--va_stf_path", default="results/egoschema/va_stf/va_stf_subset.json", type=str,
                       help="Path to the VA-STF subset data")
    args = parser.parse_args()
    main(args.filepath, args.va_stf_path)