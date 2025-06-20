# %% [markdown]
# #### 初始化全局变量，导入包

# %%
NEED_TARGET_COUNT =False
GENERATION_NUM = 5
file_tag="verifier_repair"
import json

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
# #### 查看读取sciknoweval数据

# # %%
# sciKnowEval_path_list=[]
# sciKnowEval_path_list.append(os.path.join(sciKnowEval_path, "sciknoweval_biology_test.jsonl"))
# sciKnowEval_path_list.append(os.path.join(sciKnowEval_path, "sciknoweval_chemistry_test.jsonl"))
# sciKnowEval_path_list.append(os.path.join(sciKnowEval_path, "sciknoweval_material_test.jsonl"))
# sciKnowEval_path_list.append(os.path.join(sciKnowEval_path, "sciknoweval_physics_test.jsonl"))  

# all_dfs = []
# for file_path in sciKnowEval_path_list:
#     temp_df = pd.read_json(file_path, lines=True) # 添加 lines=True
#     all_dfs.append(temp_df)

# sciKnowEval_data = pd.concat(all_dfs, ignore_index=True)

with open(os.path.join(raw_data_path, "SciKnowEval_processed_openended_filling_final_merged.json"), "r", encoding="utf-8") as f:
    sciKnowEval_data = json.load(f)
print(f"Total number of entries in SciKnowEval data: {len(sciKnowEval_data)}")
    

# %%
# print(sciKnowEval_data.iloc[52100].choices)

# %% [markdown]
# #### prompt构建

# %%
import random
def process_choices(choices):
    # 传入是"text": ["15.5 - 17.5%", "15 - 17%", "14 - 16%", "16 - 18%"], "label": ["A", "B", "C", "D"]
    # 返回的是 "A: 15.5 - 17.5%, B: 15 - 17%, C: 14 - 16%, D: 16 - 18%"
    texts = choices["text"]
    labels = choices["label"]
    if len(texts) != len(labels):
        raise ValueError("Choices and labels must have the same length.")
    formatted_choices_list=[
        [f"({label}) {text.strip()}\n" for label, text in zip(labels, texts)],
        [f"{label}: {text.strip()}\n" for label, text in zip(labels, texts)],
        [f"{label}. {text.strip()}\n" for label, text in zip(labels, texts)],
        [f"{label} - {text.strip()}\n" for label, text in zip(labels, texts)],
        [f"{label}) {text.strip()}\n" for label, text in zip(labels, texts)],
    ]
    formatted_choices = random.choice(formatted_choices_list)
    return "".join(formatted_choices)

def sciKnowEval_build_prompt(
    row,
    require_range=False              
                             ):
    # prompt = row["prompt"]["default"]
    task_type= row["type"]
    domain = row["domain"]
    subtask=row["details"]["subtask"]
    question = row["question"]
    choices = row.get("choices", None)
    true_or_false_prompt=[
        "Determine the correctness of this statement, write the correct answer inside a \\boxed{} at the end. The wrapped answer will be \"Yes\" or \"No\". ",
        "Assess the correctness of the following statement. Your conclusion, which must be either \"Yes\" or \"No\", should be placed in \\boxed{} at the end. ",
        "Evaluate the truthfulness of the statement. Your final answer should be either \"Yes\" or \"No\", enclosed in \\boxed{} at the end. ",
    ]
    choices_prompt=[
        "Answer the question, write the correct answer choice inside a \\boxed{} at the end. ",
        "Your final response should conclude with the correct answer choice wrapped in a \\boxed{}. ",
        "Please provide the answer to the question, and ensure that your final response includes the correct answer choice wrapped in a \\boxed{}. ",
    ]
    
    #task_type的类型有：
    #     "true_or_false"
    #   
    #  "mcq-2-choices"
    # "open-ended-qa"

    random_index = random.randint(0, 1)
    if task_type == "true_or_false" :
        base_prompt = "You will be presented with a hypothesis or conjecture. Based on the information provided in a text excerpt or your general knowledge, determine if the hypothesis is true (yes) or false (no). "
        prompt = random.choice(true_or_false_prompt) + "\n\n"
        format_instruction=prompt
        problem_list=[
            f"Statement: {question}\n\n",
            f"Hypothesis: {question}\n\n",
            f"Question: {question}\n\n",
        ]
        problem =base_prompt+ random.choice(problem_list)
        if random_index == 0:
            prompt= prompt+ problem
        else:
            prompt = problem + prompt
    
    elif task_type == "mcq-4-choices" or task_type == "mcq-2-choices":
        
        prompt = random.choice(choices_prompt) + "\n\n"
        format_instruction=prompt
        if choices is None:
            raise ValueError("Choices must be provided for mcq-4-choices task type.")
        formatted_choices = process_choices(choices)
        problem_list=[
            # f"Question: {qusetion}",
            f"{question}",
        ]
        problem = random.choice(problem_list)
        choice_list=[
            f"\n\n{formatted_choices}\n\n",
        ]
        problem += random.choice(choice_list)
        if random_index == 0:
            prompt = prompt + problem
        else:
            prompt = problem + prompt
    elif task_type == "open-ended-qa":
        prompt = row["prompt"]["default"]+ "\n\n"
        if subtask ==  "crystal_structure_and_composition_analysis":
            prompt="Based on the provided crystallographic data, determine the material's properties as requested in the question and list them clearly."
        elif subtask == "molecule_generation":
            prompt = "You are an expert chemist. Given a brief requirements description for molecule design, your task is to directly design a molecule, output using the SMILES of the molecule. Provide the SMILES of the molecule wrapped in a \\boxed{}. \n\n"
        
        if require_range:
            prompt += f"""
            Please provide a plausible band gap range for the new material. The width of the range should be approximately 20% of its central value. For example, for a central value of 4.0 eV, a range like '3.6 - 4.4 eV' would be appropriate.
            """
        format_instruction=prompt
        if domain == "Biology" and subtask == "text_summary":
            prompt+=question
        else:
            prompt+= f"Question: {question}"
            
        if subtask == "specified_band_gap_material_generation":
            prompt+="""Give the predicted chemical formula and bandgap values at the end in this form {"formula" : formula, "bandgap" :bandgap}"""

    elif task_type == "filling":
        prompt = "You are an expert chemist. For the given chemical equation, first provide a step-by-step reasoning on how to balance it. Then, provide the final balanced equation wrapped in a \\boxed{}. \n\n"

        format_instruction=prompt
        prompt += f"Question: {question}"
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    return prompt,format_instruction

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

# %%
from mp_api.client import MPRester
import os

API_KEY = "bSpFQg1jCJpFG4CARe0NiSUyXKke56OF"  # <--- 在这里替换成您的密钥
formula_to_search = "Li2VFe(P2O7)2"

def call_MP(formula_to_search):
    try:
        with MPRester(api_key=API_KEY) as mpr:

            # --- 步骤 1: 查询热力学性质，找到最稳定的材料 ID ---
            # 我们先从热力学端点入手，因为稳定性是首要关心的
            print("\n步骤 1: 查询热力学性质以确定最稳定的结构...")
            thermo_docs = mpr.materials.thermo.search(
                formula=formula_to_search,
                fields=["material_id", "energy_above_hull", "formula_pretty"]
            )

            if not thermo_docs:
                raise ValueError(f"在热力学数据库中未找到关于 {formula_to_search} 的材料。")

            # 按稳定性（energy_above_hull）排序，找到最稳定的那个
            stable_thermo_docs = sorted(thermo_docs, key=lambda doc: doc.energy_above_hull)
            most_stable_thermo_doc = stable_thermo_docs[0]
            
            # 获取最稳定结构的 material_id，这是我们接下来查询的“钥匙”
            stable_material_id = most_stable_thermo_doc.material_id
            energy_above_hull = most_stable_thermo_doc.energy_above_hull
            pretty_formula = most_stable_thermo_doc.formula_pretty

            print(f"找到最稳定结构 ID: {stable_material_id} (稳定性: {energy_above_hull:.3f} eV/atom)")

            # --- 步骤 2: 使用最稳定的 ID 查询电子结构性质（如带隙） ---
            print("\n步骤 2: 使用稳定结构 ID 查询电子结构性质...")
            es_doc = mpr.electronic_structure.search(
                material_ids=[stable_material_id],
                fields=["material_id", "band_gap"]
            )
            
            # es_doc 返回的是列表，我们取第一个
            band_gap = es_doc[0].band_gap if es_doc else None
            return band_gap

    except Exception as e:
        print(f"查询过程中发生错误: {e}")
        return None

# %% [markdown]
# #### 校验器

# %%
import re
from threading import Lock

import json
import json_repair
import os
import sys
import contextlib
MPLock = Lock()
def sciKnowEval_rule_verifier(answer_content: str,groundtruth: str, question: str, row=None):
    
    pattern = r"\\boxed{\\text{(.*?)}"  
    match = re.search(pattern, answer_content)

    # 如果匹配成功，提取捕获的内容
    if match:
        extracted_answer = match.group(1)
      
    else:
        return False
   
    if groundtruth.lower() == extracted_answer.lower():
        return True
    else:
        return False

def sciKnowEval_specified_band_gap_material_generation_verifier(answer_content: str,groundtruth: str, question: str) -> bool:
    from mpy_utils import predict_bandgap_for_structure
    _ ,new_formula=call_huoshan(f"Please extract the material name from the following question,Your answer should only contain the chemical formula. Do not use subscripts for numbers.\n\n content: {answer_content}","doubao")
    new_formula=new_formula.strip()
    extra_dict={}
    extra_dict["new_element"] = new_formula
    with MPLock:
        band_gap = call_MP(new_formula)
        extra_dict["database_band_gap"] = str(band_gap) if band_gap is not None else "None"
    # if not band_gap:
    if band_gap is None:
        json_str= json.dumps({
  "modification_type": "string, one of ['substitute', 'remove', 'add', 'exchange']",
  "new_material_formula": "string",
  "details": {
    "from_element": "string, only for 'substitute' or 'exchange'",
    "to_element": "string, only for 'substitute' or 'exchange'",
    "element": "string, only for 'remove' or 'add'",
    "coords": "list of floats, only for 'add'"
  }
})
        # print(f"No material found for the formula: {formula_to_search}")
        _,old_formula =call_huoshan(f"Please extract the material name from the following question,Your answer should only contain the chemical formula. Do not use subscripts for numbers.\n\n content: {question}","doubao")
        old_formula=old_formula.strip()
        extra_dict["old_element"] = old_formula
        _,parsed_json_str = call_huoshan(f"""
                                    
你是一个高精度的信息提取助手。你的任务是分析下面提供的关于材料修改的文本，并从中提取出修改类型、新材料的化学式以及修改细节。

严格按照以下 JSON 格式输出，不要包含任何额外的文字、解释或 markdown 的 ```json 标记。公式中的数字不要使用下标形式。

{json_str}

请处理以下文本：
{answer_content}
""",
"doubao")    
        with MPLock:
            parsed_json = json_repair.repair_json(parsed_json_str,return_objects=True)
            band_gap ,_= predict_bandgap_for_structure(old_formula,parsed_json)
            extra_dict["model_band_gap"] = str(band_gap) if band_gap is not None else "None"
            extra_dict["parsed_json"] = parsed_json
    prompt=f"""
You are an AI verifier specializing in material science. Your task is to determine if the model_content is a correct and scientifically valid response, considering the question and the Database Lookup Result. Output only "True" or "False".

[Context and Inputs Start]
Question: {question}
Model Content: {answer_content}
Database Lookup Result for Model's Proposed Material: band_gap: {band_gap}
[Context and Inputs End]

Evaluation Criteria:

A response is "True" if and only if ALL of the following criteria are met. If ANY criterion is not met, the response is "False".

1. Numerical Validation (Primary Rule - Must Pass):

Let database_gap be the value from the Database Lookup Result.

Let model_claimed_gap be the band gap value or range asserted by the model_content.

Define an absolute error cap: ABS_CAP = 0.3 eV.

Case A: If model_claimed_gap is a single number:

Calculate the relative tolerance: relative_tolerance = database_gap * 0.10.
Determine the allowed error by taking the stricter of the two: allowed_error = min(relative_tolerance, ABS_CAP).
Check if abs(model_claimed_gap - database_gap) <= allowed_error.
Case B: If model_claimed_gap is a range (e.g., "4.5-5.5 eV"):

Let the range be [min_val, max_val].
The response is valid only if the database_gap falls within the range, extended by the absolute error cap.
Check if (min_val - ABS_CAP) <= database_gap <= (max_val + ABS_CAP).
2. Scientific and Logical Validation (Secondary Rules - All Must Pass):

(a) Soundness of Strategy: The proposed modification must be a scientifically sound and well-reasoned strategy.

(b) Adherence to Instructions: The proposed modification must be one of the types explicitly allowed in the question.

(c) Logical Consistency: The new material proposed must be a logical and direct result of the described modification.

Strictly output "True" or "False". Do not add any explanation or other characters.

Your output:
"""
    _,llm_response=call_huoshan(prompt,"doubao")
    if llm_response.strip().lower() == "true":
        return True ,extra_dict
    elif llm_response.strip().lower() == "false":
        return False,extra_dict
    else:
        print(f"Unexpected LLM response: {llm_response}")
        return False,extra_dict

def sciKnowEval_model_verifier(answer_content: str,groundtruth: str, question: str) -> bool:
    pattern = r"\\boxed{(.*)}"
    match = re.search(pattern, answer_content)
        # 如果匹配成功，提取捕获的内容
    if match:
        # print(f"Extracted content: {match.group(1)}")
        answer_content = match.group(1)  
    else:
        # 如果不是用\\boxed{}包裹的答案，则直接使用原始内容
        answer_content = answer_content.strip()
    prompt = f"""
You are an AI verifier. Your task is to determine if the `model_content` is a correct or acceptable response to the `question`, considering the `groundtruth` as the reference for correctness. Output only "True" or "False".

[Context and Inputs Start]
Question: {question}
Ground Truth: {groundtruth}
Model Content: {answer_content}
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
 

    _,llm_response=call_huoshan(prompt,"doubao")
    if llm_response.strip().lower() == "true":
        return True
    elif llm_response.strip().lower() == "false":
        return False
    else:
        print(f"Unexpected LLM response: {llm_response}")
        return False 

def sciKnowEval_filling_verifier(answer_content: str,groundtruth: str, question: str) -> bool:
    pattern = r"\\boxed{(.*)}"
    match = re.search(pattern, answer_content)
        # 如果匹配成功，提取捕获的内容
    if match:
        # print(f"Extracted content: {match.group(1)}")
        extracted_answer = match.group(1)  
    else:
        extracted_answer = answer_content.strip()

    prompt = f"""
You are an expert chemist acting as an AI verifier. Your task is to determine if the `model_content` correctly balances the chemical equation presented in the `question`.

[Context and Inputs Start]
Question: {question}
Ground Truth (for reference): {groundtruth}
Model Content: {extracted_answer}
[Context and Inputs End]

Evaluation Criteria:

1.  Analyze the chemical equation in the `model_content`.
2.  Check if the number of atoms for each element is equal on both the reactant and product sides (i.e., the equation is correctly balanced).
3.  The `model_content` is "True" if the equation is chemically balanced. Minor formatting differences, such as using "H2" instead of "H₂" or variations in spacing, are acceptable and should be considered correct. The `ground_truth` is a reference, but the primary criterion is the correctness of the balancing itself.
4.  The `model_content` is "False" if the equation is not balanced.

Strictly output "True" or "False". Do not add any explanation or other characters.

Your output is:
"""

    _,llm_response=call_huoshan(prompt,"doubao")
    if llm_response.strip().lower() == "true":
        return True
    elif llm_response.strip().lower() == "false":
        return False
    else:
        print(f"Unexpected LLM response: {llm_response}")
        return False 


# %% [markdown]
# #### 对单个问题的处理主函数

# %%
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict

list_lock = Lock()

num_count=defaultdict(int)
def sciKnowEval_process_row(row):
    
    task_type =row["metadata"]["type"]
    prompt=row["question"]
    groundtruth = row["ground_truth"]["final_answer"]
    generations=row["generations"]
    for generation in generations:
        answer_content = generation["answer_content"]

      
        if task_type == "open-ended-qa" :
            
            correctness = sciKnowEval_model_verifier(answer_content,groundtruth,prompt)
        elif task_type == "filling":
            correctness = sciKnowEval_filling_verifier(answer_content,groundtruth,prompt)
        else:
            correctness = sciKnowEval_rule_verifier(answer_content,groundtruth,prompt)
        generation["evaluation"]["correctness"] = correctness
        



# %% [markdown]
# #### 多线程调用

# %%
import traceback
import pdb
import tqdm
res_list = []

TARGET_COUNT = 100
file_lock= Lock()
# 将结果写入JSON文件


executor = ThreadPoolExecutor(max_workers=100)
try:
   
    # 2. 在 try 块内部提交和处理任务
    counter = 0
    futures = {executor.submit(sciKnowEval_process_row, row): index for index, row in enumerate(sciKnowEval_data)}
    for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
        index = futures[future]
        try:
            future.result()
            
                
            # if counter % 1000 == 0 and counter != 0:
            #     # print(f"Processed {counter} rows.")
            #     with file_lock, list_lock:
            #         try:
            #             with open(output_file, "w", encoding="utf-8") as f:
            #                 json.dump(res_list, f, ensure_ascii=False, indent=4)
            #         except Exception as e:
            #             print(f"Error writing to file {output_file}: {e}")
            #             # pdb.set_trace()     
            
            #### 当达到目标时中断循环
            if NEED_TARGET_COUNT and counter >= TARGET_COUNT:
                print(f"已达到目标数量 {TARGET_COUNT}，中断任务循环。")
                break
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            traceback.print_exc()

finally:
    print("正在关闭线程池，不再等待剩余的慢任务...")
    executor.shutdown(wait=False, cancel_futures=True) # 这是唯一被调用的shutdown
    # try:
    #     print(f"\n任务已中断或完成，最终获取了 {len(res_list)} 个结果。")
    #     with open(output_file, "w", encoding="utf-8") as f:
    #         json.dump(res_list, f, ensure_ascii=False, indent=4)
    #     print("文件保存成功！")
    # except Exception as e:
    #     print(f"Error writing to file {output_file}: {e}")
    #     pdb.set_trace()
    


import pdb
output_file = os.path.join(raw_data_path, f"SciKnowEval_processed_{file_tag}.json")
try:
    print(f"\n任务已完成，最终获取了 {len(sciKnowEval_data)} 个结果。")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sciKnowEval_data, f, ensure_ascii=False, indent=4)
    print("文件保存成功！")
except Exception as e:
    print(f"Error writing to file {output_file}: {e}")
    pdb.set_trace()