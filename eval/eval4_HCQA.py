import os, json
import re
import pdb
from util import load_json

def build_id_dict(folder_path):
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
                    print(filename, data["ANSWER"], num, conf.group())
                    id_dict[filename.split('.')[0]] = [num, conf.group()]
                else:
                    print(filename, data["ANSWER"])

    return id_dict


# 计算准确率
def calculate_accuracy(annotations, predictions):
    correct = 0
    have_ans = 0

    for q_uid, annot in annotations.items():
        truth = annot["truth"]
        if q_uid in predictions:
            pred = predictions[q_uid]
            if truth == pred:
                correct += 1
            have_ans += 1

    accuracy = correct / have_ans if have_ans > 0 else 0
    return accuracy, have_ans

# 从 JSON 文件读取数据
annotations = load_json('data/egoschema/subset_anno.json')
folder_path = f'/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/results/egoschema/qa_by_summary/summary_interval=10_20250221_231251'  # Replace with your folder path
result_dict = build_id_dict(folder_path)
print(len(result_dict))  # 483

predictions = {}
for key, value in result_dict.items():
    predictions[key] = int(value[0])

# 计算并打印准确率
accuracy, have_ans = calculate_accuracy(annotations, predictions)
total = len(annotations)
no_ans = total - have_ans

print("Total: ", total)
print("No answer: ", no_ans)
print("have answer: ", have_ans)
print("Mean accuracy (excluded no answer): ", accuracy)
print("Mean accuracy (included no answer): ", (have_ans * accuracy + no_ans * 0.2) / total)
