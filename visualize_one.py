import os
import json
import random

# ANSI 颜色码，黄色高亮
YELLOW = '\033[93m'
RESET = '\033[0m'

def sample_entries_from_folder(file_path: str, n_per_file: int = 1):
    # 获取所有 JSON 或 JSONL 文件
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f if line.strip()]
            else:  # .json
                data = json.load(f)
    except Exception as e:
        print(f"读取 {file_path} 时出错：{e}")


    print(f"\n{'='*10} 文件: {file_path} {'='*10}\n")

    samples = random.sample(data, n_per_file)
    for idx, sample in enumerate(samples):
        print(f"{'-'*6} Sample {idx+1} {'-'*6}")
        print(f"{YELLOW}{'question'}{RESET}:\n{sample['question']}\n")
        print(f"{YELLOW}{'ground_truth'}{RESET}:\n{sample['ground_truth']['final_answer']}\n")
        #print(f"{YELLOW}{'r1'}{RESET}:\n{sample['generations'][0]['answer_content']}\n")
        #print(f"{YELLOW}{'claude'}{RESET}:\n{sample['generations'][6]['answer_content']}\n")
        input("")

# 示例调用
import os
path= os.path.dirname(os.path.abspath(__file__))
json_path= os.path.join(path,'raw_data', 'SciKnowEval_processed_choice_truefalse.json')
sample_entries_from_folder(json_path, n_per_file=10)
