import os
import json
import time
import openai
import ast
import pdb
from tqdm import tqdm
from model import GPT
from util import parse_json

systerm='''
You are a visual question answering expert. You can choose the correct answer from five options of [OPTION] based on the [CAPTION], [SUMMARY], [QUESTION], and [REASON]. Where the [CAPTION] is textual descriptions of the video as seen from your first person perspective.
'''

incontext_prompt='''
[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each caption represents a frame in the video in chronological order, although they may not be consecutive. And each description is separated by a newline character. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
[SUMMARY]: Based on the CAPTIONS of these video clips, an overall description of the video, in chronological order.
[QUESTION]: A question about video that needs to be answered.
[OPTION]: Five candidates for the question.
[REASON]: Based on [QUESTION] and [SUMMARY], reasoning step by step to get the answer. If [SUMMARY] doesn't have enough information, you need to get it from the [CAPTION].
I will give you some examples as follow:
{example}
Now, you should first make a [REASON] based on [QUESTION] and [SUMMARY], then give right number of [OPTION] as [ANSWER] . Additionally, you need to give me [CONFIDENCE] that indicates your confidence in answering the question accurately, on a scale from 1 to 5. You SHOULD answer the question, even given a low confidence.
[CAPTION]
{caption}
[SUMMARY]
{summary}
[QUESTION]
{question}
[OPTION]
{option}
'''

response_prompt='''
YOU MUST output in the JSON format.
{
"REASON":"[REASON]",
"ANSWER":"[ANSWER]",
"CONFIDENCE": "[CONFIDENCE]"
}
'''



def main():
    interval = 10
    summary_path = f"results/egoschema/summary/summary_interval={interval}"

    qa_list = os.listdir(summary_path)

    output_dir = f"results/egoschema/qa_by_summary/interval={interval}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    exist_files = os.listdir(output_dir)
    qa_list = [item for item in qa_list if item not in exist_files]
    print(len(qa_list))

    subset_anno_path = "data/egoschema/subset_anno.json"
    with open(subset_anno_path,'r') as f1:
        subset_anno = json.load(f1)

    all_cap_file = "data/egoschema/lavila_subset.json"
    all_caps = json.load(open(all_cap_file, "r"))

    example_qa_by_summary_path = "data/egoschema/example_qa_by_summary.txt"
    with open(example_qa_by_summary_path,'r') as ex:
        example = ex.read()

    api_key = "sk-mHS2kCuqAq2L1uJFE2448bDe4bCc48F8Ab880415D25a7159"
    base_url = "https://api.juheai.top/v1/"
    model_name = "gpt-4o"
    qa_model_by_summary = GPT(api_key=api_key, model_name=model_name, base_url=base_url)

    for file in tqdm(qa_list):
        # pdb.set_trace()
        uid = file[:-4]
        d = subset_anno[uid]
        try:
            captions = all_caps[uid]
            caps = ''
            for c in captions:
                caps += c + "\n"
            with open(os.path.join(summary_path, f'{uid}.txt'), 'r') as f1:
                sum = f1.read()
            opt = ''
            que = 'question: ' + d['question']
            opt += 'option 0: ' + d['option 0'] + "\n"
            opt += 'option 1: ' + d['option 1'] + "\n"
            opt += 'option 2: ' + d['option 2'] + "\n"
            opt += 'option 3: ' + d['option 3'] + "\n"
            opt += 'option 4: ' + d['option 4'] + "\n"

            instruction = str(uid) + "\n" + systerm + "\n" + incontext_prompt.format(
                example = example, caption = caps, summary = sum, question = que, option = opt
                ) + "\n" + response_prompt
            response, info = qa_model_by_summary.forward(head=None, prompt=instruction)
            
            response_dict = None
            try: 
                response_dict = ast.literal_eval(response)
            except:
                response_dict = parse_json(response)
            
            if response_dict == None:
                pdb.set_trace()

            with open(f"{output_dir}/{uid}.json", "w") as f:
                json.dump(response_dict, f)

        except openai.RateLimitError:
            print("Too many request: Sleeping 1s", flush=True)
            time.sleep(1)
        # except Exception:
        #     print(f"Error occurs when processing this query: {uid}", flush=True)
        #     break
        # else:
        #     break

if __name__ == "__main__":
    # response = '''{
    #     "REASON": "Based on the summary, the woman's actions involve various cooking-related tasks such as pouring flour into a mixer bowl, mixing flour with water, kneading dough, sieving flour, and preparing a meal by stirring and cooking vegetables. She handles multiple kitchen utensils and transfers dough across containers, indicating a focus on both cooking food and dough preparation tasks. The culminating activity of cooking vegetables in a pot strongly supports the primary objective being cooking food rather than solely making dough. Therefore, her actions align with the overall goal of cooking food.",
    #     "ANSWER": 2,
    #     "CONFIDENCE": 5
    #     }'''
    # response_dict = None
    # try: 
    #     response_dict = ast.literal_eval(response)
    # except:
    #     # response_dict = parse_json(response)
    #     response_dict = json.loads(response)
    
    # if response_dict == None:
    #     pdb.set_trace()

    main()