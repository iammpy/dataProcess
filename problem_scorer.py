# %% [markdown]
# ### 初始化全局变量，导入包

# %%
import os
import sys
from model import call_huoshan,call_openai
import pandas as pd
import json 
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

raw_data_path= os.path.join("raw_data")
data_tag="openended_filling"
sciKnowEval_path = os.path.join(raw_data_path, f"SciKnowEval_processed_openended_filling_final_merged.json")
if sciKnowEval_path.endswith(".jsonl"):
    with open(sciKnowEval_path, "r") as f:
        sciKnowEval_data= [json.loads(line) for line in f if line.strip()]
elif sciKnowEval_path.endswith(".json"):
    with open(sciKnowEval_path, "r") as f:
        sciKnowEval_data= json.load(f)
else:
    raise ValueError(f"Unsupported file format: {sciKnowEval_path}")

# %% [markdown]
# #### 处理sciKnowEval数据：生成generation，调用verifier，整合成符合要求的最终dict格式

# %%
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json_repair

Lock = Lock()
new_list= []

def sciKnowEval_process_row(row):
    
    # if len(row["generations"]) >=5:
    #     return
    # task_type = row["metadata"]["task_type"]
    id= row["id"]
    final_answer = row["ground_truth"]["final_answer"]

    question = row["question"]
    domain = row["metadata"]["domain"]
    task=row["metadata"]["details"]["task"]
    subtask = row["metadata"]["details"]["subtask"]
    ground_truth = row["ground_truth"]["final_answer"]
    
    # generations:list= row["generations"]
    generations = row["generations"] 
    prompt=f"""
You are an expert AI assistant for data quality assurance. Your task is to meticulously evaluate a given ground_truth answer based on its corresponding question.

[Input Data Start]
Question:
{question}

Ground Truth Answer:
{ground_truth}
[Input Data End]

Evaluation Instructions:

Please evaluate the Ground Truth Answer based on the following criteria, relative to the Question. Use a 0-2 scoring scale.

1. Factual Correctness (Score 0-2):
How factually accurate is the answer?
0 (Incorrect): The answer is factually wrong, logically flawed, or contains critical errors.
1 (Partially Correct): The answer is generally correct but contains minor, non-critical factual errors.
2 (Correct): The answer is perfectly accurate, with no factual errors.

2. Completeness (Score 0-2):
Does the answer address all explicit and implicit parts of the question?
0 (Incomplete): The answer completely misses the main point or ignores most parts of the question.
1 (Partially Complete): The answer addresses the main part of the question but omits significant secondary aspects.
2 (Complete): The answer comprehensively addresses all parts of the question.

3. Relevance (Score 0-2):
Is the answer focused on the question? Does it contain irrelevant information or "hallucinations"?
0 (Irrelevant): The answer is mostly irrelevant, padded with filler, or contains significant fabricated information.
1 (Mostly Relevant): The answer is generally on-topic but includes some unnecessary information that detracts from the main point.
2 (Relevant): The answer is perfectly on-topic, concise, and contains no irrelevant information.
Output Format:

Strictly output your evaluation as a single JSON object. Do not add any explanation or other characters outside of the JSON structure.

```
{{
  "factual_correctness": <score_from_0_to_2>,
  "completeness": <score_from_0_to_2>,
  "relevance": <score_from_0_to_2>,
  "justification": "<A brief, one-sentence explanation for your scoring, highlighting the main strengths or weaknesses of the answer>"
}}
```
Your Output:
    """
    
    _,json_str=call_huoshan(prompt,"doubao")
    json_ans= json_repair.repair_json(json_str,return_objects=True)
    new_list.append({
        "id": id,
        "domain": domain,
        "task": task,
        "subtask": subtask,
        "question": question,
        "ground_truth": ground_truth,
        "question_quality": json_ans["question_quality"],
        "answer_correctness": json_ans["answer_correctness"],
        "justification": json_ans["justification"]
    })
    
    return 0
        


# %%
import traceback
import pdb
import tqdm
num=0
with ThreadPoolExecutor(max_workers=400) as executor:
    counter = 0
    futures = {executor.submit(sciKnowEval_process_row, row): index for index, row in enumerate(sciKnowEval_data)}
    for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
        index = futures[future]
        return_num= future.result()
        # if return_num ==0:
        #     num+=1
        #     # print(f"成功修复了 {num} 个数据。")
        # try:
        #     future.result()  # 获取结果，确保异常被捕获
        #     counter += 1
        #     # if counter % 10 == 0:
        #         # print(f"Processed {counter} rows.")
        # except Exception as e:
        #     print(f"Error processing row {index}: {e}")
        #     traceback.print_exc() 



# %%
score_path = os.path.join(raw_data_path, f"SciKnowEval_processed_openended_score.json")
try:
    with open(score_path, "w", encoding="utf-8") as f:
        
        json.dump(new_list, f, ensure_ascii=False, indent=4)

except Exception as e:
    print(f"Error writing to file {score_path}: {e}")
    pdb.set_trace()
    hash("abc")
    



