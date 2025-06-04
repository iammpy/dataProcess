# %% [markdown]
# ### 初始化全局变量，导入包

# %%
import os
import sys
from model import call_huoshan,call_openai
import pandas as pd
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

raw_data_path= os.path.join("raw_data")
scienceQA_path = os.path.join(raw_data_path, "ScienceQA")
sciKnowEval_path = os.path.join(raw_data_path, "SciKnowEval")

# %% [markdown]
# ### 查看并读取sciQA数据

# %%
# filepath: /u01/mengpengyu/dataProcess/sciknowevalProcess.ipynb

sciQA_path=[]
sciQA_path.append(os.path.join(scienceQA_path, "test-00000-of-00001-f0e719df791966ff.parquet"))
sciQA_path.append(os.path.join(scienceQA_path, "train-00000-of-00001-1028f23e353fbe3e.parquet"))
sciQA_path.append(os.path.join(scienceQA_path, "validation-00000-of-00001-6c7328ff6c84284c.parquet"))

all_dfs = []
for file_path in sciQA_path:
    temp_df = pd.read_parquet(file_path)
    all_dfs.append(temp_df)

sciQA_data = pd.concat(all_dfs, ignore_index=True)


# %% [markdown]
# ### 查看读取sciknoweval数据

# %%
sciKnowEval_path_list=[]
sciKnowEval_path_list.append(os.path.join(sciKnowEval_path, "sciknoweval_biology_test.jsonl"))
sciKnowEval_path_list.append(os.path.join(sciKnowEval_path, "sciknoweval_chemistry_test.jsonl"))
sciKnowEval_path_list.append(os.path.join(sciKnowEval_path, "sciknoweval_material_test.jsonl"))
sciKnowEval_path_list.append(os.path.join(sciKnowEval_path, "sciknoweval_physics_test.jsonl"))  

all_dfs = []
for file_path in sciKnowEval_path_list:
    temp_df = pd.read_json(file_path, lines=True) # 添加 lines=True
    all_dfs.append(temp_df)

sciKnowEval_data = pd.concat(all_dfs, ignore_index=True)

# %% [markdown]
# #### 根据一个问题，以及不同的文件类型，构建传给模型的最终prompt

# %%
def process_choices(choices):
    # 传入是"text": ["15.5 - 17.5%", "15 - 17%", "14 - 16%", "16 - 18%"], "label": ["A", "B", "C", "D"]
    # 返回的是 "A: 15.5 - 17.5%, B: 15 - 17%, C: 14 - 16%, D: 16 - 18%"
    texts = choices["text"]
    labels = choices["label"]
    if len(texts) != len(labels):
        raise ValueError("Choices and labels must have the same length.")
    formatted_choices = [f"({label}) {text}" for label, text in zip(labels, texts)]
    return " ".join(formatted_choices)

# test= {
#     "text": ["15.5 - 17.5%", "15 - 17%", "14 - 16%", "16 - 18%"], 
#     "label": ["A", "B", "C", "D"]
# }
# print(process_choices(test))

def sciKnowEval_build_prompt(row):
    prompt = row["prompt"]["default"]
    task_type= row["type"]
    qusetion = row["question"]
    choices = row.get("choices", None)
    #task_type的类型有：
    #     "true_or_false"
    #   
    #  "mcq-2-choices"
    # "open-ended-qa"
    if task_type == "true_or_false" or task_type == "open-ended-qa":
        prompt += f"Question: {qusetion}\n\n"
    
    elif task_type == "mcq-4-choices" or task_type == "mcq-2-choices":
        if choices is None:
            raise ValueError("Choices must be provided for mcq-4-choices task type.")
        formatted_choices = process_choices(choices)
        prompt += f"Question: {qusetion}"
        prompt += f"\n\nChoices: {formatted_choices}"

    else:
        raise ValueError(f"Unknown task type: {task_type}")
    return prompt

# %%
import hashlib

def generate_md5(input_string):
    # 创建一个 md5 hash 对象
    md5_hash = hashlib.md5()
    
    # 将输入的字符串转换为字节串（因为 hashlib 需要字节类型的数据）
    input_bytes = input_string.encode('utf-8')
    
    # 更新哈希对象
    md5_hash.update(input_bytes)
    
    # 获取哈希值的十六进制表示
    md5_digest = md5_hash.hexdigest()
    
    return md5_digest

# 示例使用
input_string = "Hello, World!"
md5_result = generate_md5(input_string)
print(f"MD5 of '{input_string}': {md5_result}")

# %% [markdown]
# #### sciKnowEval的验证器，openEnded问题使用模型验证，其余使用规则直接比对

# %%
def sciKnowEval_rule_verifier(question, groundtruth, model_content):
    if groundtruth == model_content:
        return True
    else:
        return False
def sciKnowEval_model_verifier(question: str, groundtruth: str, model_content: str) -> bool:

    prompt = f"""
You are an AI verifier. Your task is to determine if the `model_content` is a correct or acceptable response to the `question`, considering the `groundtruth` as the reference for correctness. Output only "True" or "False".

[Context and Inputs Start]
Question: {question}
Ground Truth: {groundtruth}
Model Content: {model_content}
[Context and Inputs End]

Evaluation Criteria:

Compare the `model_content` with the `groundtruth` in the context of the `question`.

"The model_content is "True" if it proposes a scientifically sound and well-reasoned modification to the starting material, correctly applying one of the specified modification types from the question, and the proposed new material is a logical outcome of this modification. The rationale should clearly explain how this modification is expected to lead towards the target property. The groundtruth serves as a reference for a potentially valid outcome, but a well-argued alternative solution that also meets the question's constraints and scientific principles is also considered "True"."

The model_content is "False" if it:
Fails to apply a valid modification type as specified in the question to the correct starting material.
Contains critical scientific flaws in its reasoning or proposed modification.
The proposed new material is not a logical or direct result of the described modification process.
Fundamentally misunderstands the scientific goal or constraints of the question.

Strictly output "True" or "False". Do not add any explanation or other characters.

Your output is:
"""

    _,llm_response=call_openai(prompt)
    if llm_response.strip().lower() == "true":
        return True
    elif llm_response.strip().lower() == "false":
        return False
    else:
        print(f"Unexpected LLM response: {llm_response}")
        return False 


# %% [markdown]
# #### 处理sciKnowEval数据：生成generation，调用verifier，整合成符合要求的最终dict格式

# %%
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

Lock = Lock()


def sciKnowEval_process_row(row):
    global res_list
    task_type = row["type"]
    if task_type == "mcq-2-choices" or task_type == "mcq-4-choices":
        groundtruth = row["answerKey"]
    elif task_type == "open-ended-qa" or task_type == "true_or_false":
        groundtruth = row["answer"]

    prompt = sciKnowEval_build_prompt(row)
    
    generations=[]
    for i in range(1): # 调用模型的次数，暂定为1
        generation={}
        generation["model"] = "DeepSeek-R1"
        reasoning_content, answer_content = call_huoshan(prompt,"r1")
        answer_content=answer_content.strip()
        generation["reasoning_content"] = reasoning_content
        generation["answer_content"] = answer_content
        # Verify the model content
        evaluation={}
        if task_type == "open-ended-qa":
            correctness = sciKnowEval_model_verifier(prompt, groundtruth, answer_content)
        else:
            correctness = sciKnowEval_rule_verifier(prompt, groundtruth, answer_content)
        evaluation["correctness"] = correctness
        evaluation["By"] = "mengpengyu"
        evaluation["Method"] = "gpt-4o" if task_type == "open-ended-qa" else "Rule"
        evaluation["extra_tags"] = []
        generation["evaluation"] = evaluation
        generations.append(generation)
        
    if task_type == "mcq-2-choices" or task_type == "mcq-4-choices":
        task_type= "multiple_choice_single"
    elif task_type == "open-ended-qa":
        task_type= "question_answering" 
        
    res_dict={}
    res_dict["id"] = generate_md5(prompt)
    res_dict["metadata"] = row.to_dict()
    res_dict["source_dataset"] = "hicai-zju/SciKnowEval"
    # res_dict["subject_info"] = row["domain"]   #待定，额外对数据进行打标？
    res_dict["task_type"] = task_type
    res_dict["languages"] = "en"
    res_dict["multimedia"]= []
    res_dict["question"] = prompt
    res_dict["ground_truth"] = {
            "final_answer": groundtruth,
            "unit": None, 
            "solution": None,
            "extra_tags": []
        }
    res_dict["generations"]=generations
    res_dict["solve_rate"] = sum(1 for gen in generations if gen["evaluation"]["correctness"]) / len(generations)
    res_dict["prompted_for_correct_answer"]= False
    with Lock:
        res_list.append(res_dict)
    


# %%
import traceback
res_list = []
with ThreadPoolExecutor(max_workers=100) as executor:
    counter = 0
    futures = {executor.submit(sciKnowEval_process_row, row): index for index, row in sciKnowEval_data.iloc[:].iterrows()}
    for future in as_completed(futures):
        index = futures[future]
        try:
            future.result()  # 获取结果，确保异常被捕获
            counter += 1
            if counter % 10 == 0:
                print(f"Processed {counter} rows.")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            traceback.print_exc() 
# 将结果写入JSON文件
output_file = os.path.join(raw_data_path, "SciKnowEval_processed.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(res_list, f, ensure_ascii=False, indent=4)



