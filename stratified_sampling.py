import json
import random
import os
import math
from collections import Counter

def stratified_sampling(data_name):
    try:
        # 读取dirty数据文件
        print(f"正在读取{data_name}")
        with open(f"{data_name}.json", "r", encoding="utf-8") as f:
            dirty_data = json.load(f)
        
        total_samples = len(dirty_data)
        print(f"读取到 {total_samples} 条被拒绝的数据")
        
        # 统计各拒绝类型的数量和比例
        reject_types = [item.get("reject_type", "未知") for item in dirty_data]
        type_counter = Counter(reject_types)
        
        # 按数量从高到低排序
        sorted_types = sorted(type_counter.items(), key=lambda x: x[1], reverse=True)
        
        # 输出拒绝类型分布
        print("\n拒绝类型分布:")
        print("-" * 50)
        print(f"{'拒绝类型':<30} | {'数量':<10} | {'百分比':<10}")
        print("-" * 50)
        
        for reject_type, count in sorted_types:
            percentage = (count / total_samples) * 100
            print(f"{reject_type:<30} | {count:<10} | {percentage:.2f}%")
        
        # 获取用户输入的采样总数
        while True:
            try:
                sample_total = int(input("\n请输入采样总数: "))
                if sample_total <= 0:
                    print("采样总数必须大于0，请重新输入")
                    continue
                if sample_total > total_samples:
                    print(f"采样总数不能大于总样本数 {total_samples}，请重新输入")
                    continue
                break
            except ValueError:
                print("请输入有效的整数")
        
        # 根据比例计算每种类型需要采样的数量
        sample_counts = {}
        remaining = sample_total
        processed_types = 0
        for reject_type, count in sorted_types:
            if processed_types == len(sorted_types) - 1:
                # 最后一种类型，分配所有剩余样本以避免舍入误差
                sample_counts[reject_type] = remaining
            else:
                # 根据比例计算样本数，使用math.floor确保不会超过总数
                type_ratio = count / total_samples
                type_samples = math.floor(sample_total * type_ratio)
                sample_counts[reject_type] = type_samples
                remaining -= type_samples
            processed_types += 1
        
        # 确认采样计划
        print("\n采样计划:")
        print("-" * 50)
        print(f"{'拒绝类型':<30} | {'原始数量':<10} | {'采样数量':<10} | {'采样比例':<10}")
        print("-" * 50)
        
        total_planned = 0
        for reject_type, count in sorted_types:
            planned_count = sample_counts[reject_type]
            total_planned += planned_count
            sampling_ratio = (planned_count / count) * 100 if count > 0 else 0
            print(f"{reject_type:<30} | {count:<10} | {planned_count:<10} | {sampling_ratio:.2f}%")
        
        print(f"\n计划采样总数: {total_planned}")
        confirm = input("确认执行采样? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("已取消采样")
            return
        
        # 执行采样并保存结果
        # 先按拒绝类型对数据进行分组
        type_groups = {}
        for item in dirty_data:
            reject_type = item.get("reject_type", "未知")
            if reject_type not in type_groups:
                type_groups[reject_type] = []
            type_groups[reject_type].append(item)
        
        # 创建输出目录
        output_dir = "stratified_samples"
        os.makedirs(output_dir, exist_ok=True)
        
        # 对每种类型进行采样并保存
        for reject_type, items in type_groups.items():
            if reject_type not in sample_counts or sample_counts[reject_type] <= 0:
                continue
            
            # 确定要采样的数量
            sample_count = min(sample_counts[reject_type], len(items))
            if sample_count == 0:
                continue
            
            # 随机采样
            sampled_items = random.sample(items, sample_count)
            
            # 保存到文件
            # 使用拒绝类型作为文件名，替换可能的非法字符
            safe_filename = reject_type.replace("/", "_").replace("\\", "_")
            safe_filename = safe_filename.replace(":", "_").replace("*", "_")
            safe_filename = safe_filename.replace("?", "_").replace("\"", "_")
            safe_filename = safe_filename.replace("<", "_").replace(">", "_")
            safe_filename = safe_filename.replace("|", "_")
            
            output_file = os.path.join(output_dir, f"{safe_filename}.jsonl")
            
            with open(output_file, "w", encoding="utf-8") as f:
                for item in sampled_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"已将 {sample_count} 条 '{reject_type}' 类型的样本保存至 {output_file}")
        
        print("\n采样完成!")
        print(f"所有样本文件已保存在 {output_dir} 目录中")
    
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "raw_data"))
    data_name="SciKnowEval_processed_openended_filling_dirty"
    stratified_sampling(data_name) 