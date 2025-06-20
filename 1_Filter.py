import json
import yaml
import requests
import time
from tqdm import tqdm
import concurrent.futures
import json_repair
import os
import random
import glob

os.chdir(os.path.join(os.path.dirname(__file__),'raw_data'))

# 读取配置文件
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('doubao-1.6-thinking-pro', {})
#doubao-1.5-thinking-pro
#deepseek-r1
def call_deepseek_api(prompt, config):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }
    
    data = {
        "model": config['model_name'],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config['temperature'],
        "top_p": config['top_p'],
        "seed": config['seed']
    }
    
    retry_count = 0
    while retry_count < config['max_retries']:
        try:
            response = requests.post(
                config['base_url'],
                headers=headers,
                json=data,
                timeout=config['timeout']
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"API 调用失败 ({retry_count+1}/{config['max_retries']}): {str(e)}")
            retry_count += 1
            time.sleep(config['retry_delay'])
    
    return None

# 构建提示词
def create_prompt(question, answer, question_type):
    return f"""你是一名问题质量评估专家。请根据下文评估标准，严格评估给定问题是否存在表述不清、信息缺失或答案歧义的问题。

【评估标准】
1. 上下文缺失：问题依赖前文、图示等信息，但这些信息没有随题给出，导致问题在脱离上下文后无法解答。负例："根据下图，计算物体所受的合力大小。选项：A. 10N B. 5N"（题目中未提供任何力学分析图，导致无法解答）
2. 选择题选项缺失：题型为选择题"multiple_choice_xx"，但未列出选项。负例："Which of the following physical quantities are vectors? (The question does not list specific options)"
3. 代词指代不明：问题中的代词（如it，this，that）指代对象不明确，上下文中有多个可能的指代对象，导致歧义。负例："Given two reactions: Zn(s) + 2HCl(aq) → ZnCl₂(aq) + H₂(g) and 2H₂(g) + O₂(g) → 2H₂O(l), what is the change in heat of this reaction? A. Exothermic B. Endothermic"（this reaction指哪一个反应？）
4. 术语存在歧义：问题中使用的专业术语在当前语境下有多种常见解释，无法确定其唯一含义。负例："Please calculate the system's efficiency." (The definition of "efficiency" is unclear, as it could mean thermal efficiency, mechanical efficiency, or quantum efficiency.)
5. 选择题选项重复：选择题"multiple_choice_xx"中，不同选项的含义本质上相同，导致答案不唯一。负例："What is the chemical formula of vinegar? A. H3CC(=O)OH B. CH3COOH"(A选项和B选项等价，同是醋酸的化学式)
6. 约束条件不完整：题目缺少推导唯一答案所必需的关键条件（如反应条件、物质状态、参数范围），此类缺失无法依靠常识自动补足。负例："将5g某金属与足量稀盐酸反应，计算生成氢气的质量。"（未指明金属种类（如Mg、Al、Zn），不同金属摩尔质量和反应价态不同，导致氢气质量无法唯一确定）
7. 开放或主观问题：当问题类型为"question_answering"时，若问题是开放式的，并且题目中未设置备选选项或限定条件限制答案范围，会导致答案发散，无法基于客观科学依据得出唯一结论。负例："请讨论气候变化的影响。"（问题应添加限定条件，如"对北极地区哺乳动物的影响"、"在未来50年内的影响"等）

【注意事项】
- 你不需要验证问题的选项和参考答案的正确性，只需要依据"评估标准 1-7"检查问题是否属于其中一项。
- 问题类型为"multiple_choice_single"（单选题）时，只要题目足以选出最优选项，即使未显式写出常见假设（如等压、等温等），也不应因此拒绝。
- 问题中不提供公认的科学常数（如氮原子质量14g/mol等），不视为"约束条件不完整"。
- 仅当题目违反"评估标准 1-7"且常识无法补救时，才标记 REJECT，若问题虽有小瑕疵但仍能明确得出唯一答案，应标记 PASS。

【输入内容】
问题：
{question}

参考答案：
{answer}

问题类型：
{question_type}

【你的任务】
请阅读问题、参考答案和问题类型，参考评估标准 1-7，判断问题是否合格，并生成符合"输出格式"要求的 JSON 结果。

【输出格式】
- 仅返回一个合法 JSON 对象，不得包含任何额外文字。
- 字段含义：
  - "reason": 拒绝原因的描述；若通过则为空字符串。
  - "type": 最主要的拒绝类型（如 "上下文缺失"、"选择题选项缺失" 等）；若通过则为空字符串。
  - "result": "PASS" 或 "REJECT"。
- 示例：
{{
  "reason": "",
  "type": "",
  "result": "PASS/REJECT"
}}
"""

# 尝试修复损坏的JSON
def repair_json(json_str):
    try:
        # 尝试解析修复后的JSON
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 如果无法修复，返回默认格式
        answer = json_repair.loads(json_str)
        if answer is None:
            return {"result": "REJECT", "type": "-", "reason": "无法解析API返回内容"}
        else:
            return answer

# 处理单个条目的函数
def process_item(item, config):
    question = item.get("question", "")
    answer = ""
    
    # 获取答案
    if "ground_truth" in item and "final_answer" in item["ground_truth"]:
        answer = item["ground_truth"]["final_answer"]
    
    # 修复答案判断逻辑：避免将数字0误判为空
    # 检查问题是否为空或只包含空白字符
    if not question or not question.strip():
        print(f"question为空: {repr(question)}")
        item["reject_reason"] = "问题为空"
        item["reject_type"] = "-"
        return "dirty", item
    
    # 检查答案是否为None或空字符串（但数字0是有效答案）
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        print(f"answer为空: {repr(answer)}")
        item["reject_reason"] = "答案为空"
        item["reject_type"] = "-"
        return "dirty", item
    
    if "type" in item:
        question_type = item["type"]
    else:
        question_type = "未知"
        
    # 构建提示词并调用API
    prompt = create_prompt(question, answer, question_type)
    response = call_deepseek_api(prompt, config)
    
    # 处理API响应
    if response is None:
        item["reject_reason"] = "API调用失败"
        item["reject_type"] = "-"
        return "dirty", item
    
    # 尝试解析JSON返回结果
    try:
        result_json = json.loads(response)
    except json.JSONDecodeError:
        # 如果解析失败，尝试修复JSON
        print(f"JSON解析失败，尝试修复: {response[:100]}...")
        result_json = repair_json(response)
    
    # 确保结果包含所有必需字段
    if not isinstance(result_json, dict) or "result" not in result_json:
        item["reject_reason"] = f"API返回格式错误: {response[:100]}..."
        item["reject_type"] = "-"
        return "dirty", item
    
    # 处理评估结果
    if result_json.get("result") == "PASS":
        return "clean", item
    else:
        # 获取拒绝类型和原因
        reject_type = result_json.get("type", "-")
        reject_reason = result_json.get("reason", "未提供拒绝原因")
        
        item["reject_type"] = reject_type
        item["reject_reason"] = reject_reason
                
        return "dirty", item

# 保存单个batch的结果
def save_batch_results(batch_clean_items, batch_dirty_items, clean_file_prefix, dirty_file_prefix, batch_idx):
    """保存单个batch的结果"""
    # 创建batch文件名
    clean_batch_file = f"{clean_file_prefix}_batch_{batch_idx:04d}.json"
    dirty_batch_file = f"{dirty_file_prefix}_batch_{batch_idx:04d}.json"
    
    # 保存清洁数据
    with open(clean_batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch_clean_items, f, ensure_ascii=False, indent=2)
    
    # 保存不干净数据
    with open(dirty_batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch_dirty_items, f, ensure_ascii=False, indent=2)
    
    print(f"批次 {batch_idx + 1} 已保存: 干净 {len(batch_clean_items)} 条, 不干净 {len(batch_dirty_items)} 条")
    return clean_batch_file, dirty_batch_file

# 合并所有batch文件
def merge_all_batches(clean_file_prefix, dirty_file_prefix, final_clean_file, final_dirty_file):
    """合并所有batch文件到最终文件"""
    print("正在合并所有batch文件...")
    
    # 查找所有batch文件
    clean_batch_files = sorted(glob.glob(f"{clean_file_prefix}_batch_*.json"))
    dirty_batch_files = sorted(glob.glob(f"{dirty_file_prefix}_batch_*.json"))
    
    # 合并清洁数据
    all_clean_items = []
    for batch_file in clean_batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                all_clean_items.extend(batch_data)
            print(f"已合并 {batch_file}: {len(batch_data)} 条记录")
        except Exception as e:
            print(f"读取 {batch_file} 时出错: {e}")
    
    # 合并不干净数据
    all_dirty_items = []
    for batch_file in dirty_batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                all_dirty_items.extend(batch_data)
            print(f"已合并 {batch_file}: {len(batch_data)} 条记录")
        except Exception as e:
            print(f"读取 {batch_file} 时出错: {e}")
    
    # 保存合并后的最终文件
    with open(final_clean_file, 'w', encoding='utf-8') as f:
        json.dump(all_clean_items, f, ensure_ascii=False, indent=2)
    
    with open(final_dirty_file, 'w', encoding='utf-8') as f:
        json.dump(all_dirty_items, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成！")
    print(f"总计干净数据: {len(all_clean_items)} 条 -> {final_clean_file}")
    print(f"总计不干净数据: {len(all_dirty_items)} 条 -> {final_dirty_file}")
    
    return len(all_clean_items), len(all_dirty_items)

# 主函数
def filter(file_name=None,
           file_type="json"
           ):
    # 加载配置
    print("正在加载配置文件...")
    config = load_config("api_config.yaml")
    
    # file_name=f"Chembench_问题改写"
    # 打开输入和输出文件
    input_file = f"{file_name}.{file_type}"
    clean_file = f"{file_name}_clean.json"
    dirty_file = f"{file_name}_dirty.json"
    
    # 设置batch文件前缀
    clean_file_prefix = os.path.splitext(clean_file)[0]
    dirty_file_prefix = os.path.splitext(dirty_file)[0]
    
    print(f"正在读取{input_file}...")
    # input_file = os.path.join(os.getcwd(), input_file)
    #读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.jsonl'):
            # 如果是JSON Lines格式，逐行读取
            items = [json.loads(line) for line in f if line.strip()]
        elif input_file.endswith('.json'):
            # 如果是标准JSON格式，直接加载
            items = json.load(f)
        else:
            raise ValueError("不支持的文件格式，请提供 .json 或 .jsonl 文件")
        
    print(f"共读取到 {len(items)} 条记录")
    """
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    """
    # 设置并发处理的线程数
    max_workers = config.get('max_workers', 200)
    print(f"使用 {max_workers} 个线程进行并发处理...")

    # 设置批处理大小
    batch_size = 100000
    
    # 计算需要处理的批次数
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    # 存储所有batch文件路径
    batch_files = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(items))
        batch_items = items[start_idx:end_idx]
        
        print(f"正在处理第 {batch_idx + 1}/{total_batches} 批 (数据 {start_idx+1}-{end_idx})")
        
        # 准备存储当前batch结果的列表
        batch_clean_items = []
        batch_dirty_items = []
        
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_item, item, config) for item in batch_items]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(batch_items), desc=f"批次 {batch_idx + 1}"):
                batch_results.append(future.result())
        
        # 处理批次结果
        for status, item_data in batch_results:
            if status == "clean":
                batch_clean_items.append(item_data)
            else:
                batch_dirty_items.append(item_data)
        
        # 保存当前batch的结果
        clean_batch_file, dirty_batch_file = save_batch_results(
            batch_clean_items, batch_dirty_items, 
            clean_file_prefix, dirty_file_prefix, batch_idx
        )
        batch_files.append((clean_batch_file, dirty_batch_file))
    
    # 合并所有batch文件
    total_clean, total_dirty = merge_all_batches(
        clean_file_prefix, dirty_file_prefix, clean_file, dirty_file
    )
    
    print(f"处理完成！共处理 {len(items)} 条记录")
    print(f"通过: {total_clean} 条")
    print(f"拒绝: {total_dirty} 条")
    print(f"干净数据已保存至: {clean_file}")
    print(f"不干净数据已保存至: {dirty_file}")
    base_path = "."
    from move_and_merge import merge
    merge(base_path=base_path, file_name=file_name)

if __name__ == "__main__":
    file_name_list=[
        # "AGIEval_问题改写",
        # "Chembench_问题改写",
    ]
    jsonl_file_name_list=[
    #    "SciKnowEval_processed_repair",
    # "SciKnowEval_processed_chemical_filling",
    # "SciKnowEval_processed_molecule_generation",
    # "SciKnowEval_processed_text_summary",
    "SciKnowEval_processed_repair",
    
    ]
    for file_name in file_name_list:
        filter(file_name)
    for file_name in jsonl_file_name_list:
        filter(file_name, file_type="jsonl")


