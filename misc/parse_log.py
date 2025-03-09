import os
import re

def split_log_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    video_records = []
    current_record = []
    previous_line = ""
    for line in lines:
        if "[{'role': 'system', 'content': 'You are a helpful assistant.'}" in line:
            if "- util - INFO - Step" not in previous_line:
                if current_record:
                    video_records.append(current_record)
                    current_record = []
        current_record.append(line.strip())
        previous_line = line.strip()

    if current_record:
        video_records.append(current_record)

    return video_records

def classify_records(records):
    error_records = []
    success_records = []
    for record in records:
        record_str = "\n".join(record)
        if "- util - ERROR -" in record_str:
            error_records.append(record)
        else:
            success_records.append(record)
    return error_records, success_records

def save_records(records, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for i, record in enumerate(records):
        with open(f"{output_dir}/{prefix}_video_record_{i+1}.log", 'w') as file:
            file.write("\n".join(record))

def calculate_accuracy(success_records):
    total = len(success_records)
    correct = 0
    pattern = re.compile(r"Finished video: (.+)/(\d+)/(\d+)")
    for record in success_records:
        for line in record:
            match = pattern.search(line)
            if match:
                video_id, pred, ans = match.groups()
                if pred == ans:
                    correct += 1
                break
    accuracy = correct / total if total > 0 else 0
    return accuracy

if __name__ == "__main__":
    log_filepath = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/results/egoschema/ta/ta_subset_20250217_212338.log"
    # output_directory = "/Users/sunqifan/Documents/codes/video_agents/TreeVideoAgent/results/egoschema/ta/split_logs"

    video_records = split_log_file(log_filepath)
    error_records, success_records = classify_records(video_records)

    # save_records(error_records, output_directory, "error")
    # save_records(success_records, output_directory, "success")

    accuracy = calculate_accuracy(success_records)
    print(f"Processed {len(video_records)} video records.")
    print(f"Error records: {len(error_records)}")
    print(f"Success records: {len(success_records)}")
    print(f"Accuracy of successful records: {accuracy:.2%}")

    acc_all = (accuracy * len(success_records) + 0.25 * len(error_records)) / len(video_records)
    print(f"Overall accuracy: {acc_all:.2%}")