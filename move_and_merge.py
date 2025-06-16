import json
import os
from pathlib import Path

def find_deepseek_r1_value(solve_rate_dict):
    """
    在solve_rate字典中查找DeepSeek-R1的值，支持多种键名变体
    
    Args:
        solve_rate_dict: 包含solve_rate信息的字典
        
    Returns:
        float: DeepSeek-R1的值，如果找不到则返回0
    """
    if not solve_rate_dict:
        return 0
    
    # 定义可能的键名变体
    possible_keys = [
        "DeepSeek-R1",
        "deepseek-r1", 
        "Deepseek-r1",
        "deepSeek-R1",
        "DeepSeek-r1",
        "DEEPSEEK-R1",
        "deepseek-R1",
        "Deepseek-R1"
    ]
    
    # 遍历所有可能的键名
    for key in possible_keys:
        if key in solve_rate_dict:
            return solve_rate_dict[key]
    
    # 如果以上都没找到，尝试不区分大小写的查找
    for key in solve_rate_dict.keys():
        if key.lower().replace("-", "").replace("_", "") == "deepseekr1":
            return solve_rate_dict[key]
    
    return 0

def should_restore_item(item):
    """
    判断一个dirty项目是否应该被还原到clean中
    
    Args:
        item: 数据项
        
    Returns:
        bool: 是否应该还原
    """
    reject_type = item.get("reject_type")
    
    # 检查reject_type是否为"上下文缺失"
    if reject_type == "上下文缺失":
        if item.get("type") == "multiple_choice_single":
            solve_rate = item.get("solve_rate", {})
            deepseek_r1_value = find_deepseek_r1_value(solve_rate)
            return deepseek_r1_value > 0.2
        else:
            solve_rate = item.get("solve_rate", {})
            deepseek_r1_value = find_deepseek_r1_value(solve_rate)
            return deepseek_r1_value > 0
    
    # 检查其他reject_type
    if reject_type in ["约束条件不完整", "开放或主观问题"]:
        solve_rate = item.get("solve_rate", {})
        deepseek_r1_value = find_deepseek_r1_value(solve_rate)
        return deepseek_r1_value > 0.2
    
    return False

def process_files(clean_file_path, dirty_file_path, output_file_path):
    """
    处理clean和dirty文件，应用move规则并合并输出
    
    Args:
        clean_file_path: clean文件路径
        dirty_file_path: dirty文件路径
        output_file_path: 输出文件路径
    """
    try:
        # 读取clean文件
        print(f"正在读取clean文件: {clean_file_path}")
        if not os.path.exists(clean_file_path):
            print(f"错误：clean文件不存在: {clean_file_path}")
            return False
            
        with open(clean_file_path, "r", encoding="utf-8") as f:
            clean_data = json.load(f)
        print(f"读取到 {len(clean_data)} 条clean数据")
        
        # 读取dirty文件
        print(f"正在读取dirty文件: {dirty_file_path}")
        if not os.path.exists(dirty_file_path):
            print(f"错误：dirty文件不存在: {dirty_file_path}")
            return False
            
        with open(dirty_file_path, "r", encoding="utf-8") as f:
            dirty_data = json.load(f)
        print(f"读取到 {len(dirty_data)} 条dirty数据")
        
        # 为clean数据添加data_filter字段（包含raw信息）
        print("正在为clean数据添加data_filter字段...")
        for item in clean_data:
            # 提取原始评估信息（clean数据通常没有reject字段，所以是PASS状态）
            raw_info = {
                "reason": item.get("reject_reason", ""),
                "type": item.get("reject_type", ""),
                "result": "PASS"  # clean数据默认为PASS
            }
            
            # 添加data_filter字段
            item["data_filter"] = {
                "label": "PASS",
                "reject_type": "",
                "reject_reason": "",
                "raw": raw_info
            }
        
        # 为dirty数据添加data_filter字段（包含raw信息）
        print("正在为dirty数据添加data_filter字段...")
        for item in dirty_data:
            # 提取原始评估信息
            raw_info = {
                "reason": item.get("reject_reason", ""),
                "type": item.get("reject_type", ""),
                "result": "REJECT"  # dirty数据默认为REJECT
            }
            
            # 添加data_filter字段
            item["data_filter"] = {
                "label": "REJECT",
                "reject_type": item.get("reject_type", ""),
                "reject_reason": item.get("reject_reason", ""),
                "raw": raw_info
            }
        
        # 应用move规则，分离需要还原的数据和保留在dirty的数据
        restored_data = []  # 从dirty还原到clean的数据
        remaining_dirty_data = []  # 仍然保留在dirty的数据
        restored_count = 0
        
        for item in dirty_data:
            if should_restore_item(item):
                # 创建副本并更新data_filter状态为PASS
                item_copy = item.copy()
                
                # 更新data_filter字段（保留raw信息，但更新label）
                if "data_filter" in item_copy:
                    item_copy["data_filter"]["label"] = "PASS"
                    item_copy["data_filter"]["reject_type"] = ""
                    item_copy["data_filter"]["reject_reason"] = ""
                    # raw信息保持不变
                
                # 移除原有的reject字段（如果存在）
                if "reject_type" in item_copy:
                    del item_copy["reject_type"]
                if "reject_reason" in item_copy:
                    del item_copy["reject_reason"]
                    
                restored_data.append(item_copy)
                restored_count += 1
            else:
                remaining_dirty_data.append(item)
        
        print(f"根据move规则，从dirty中还原了 {restored_count} 条数据")
        print(f"剩余dirty数据: {len(remaining_dirty_data)} 条")
        
        # 合并数据
        merged_data = []
        
        # 添加原始clean数据
        merged_data.extend(clean_data)
        
        # 添加还原的数据
        merged_data.extend(restored_data)
        
        # 添加剩余的dirty数据
        merged_data.extend(remaining_dirty_data)
        
        # 保存合并结果
        print(f"正在保存合并结果到: {output_file_path}")
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        # 统计信息
        total_pass = len(clean_data) + restored_count
        total_reject = len(remaining_dirty_data)
        total_items = total_pass + total_reject
        
        print(f"\n处理完成！")
        print(f"总数据量: {total_items} 条")
        print(f"  原始clean: {len(clean_data)} 条")
        print(f"  还原数据: {restored_count} 条")
        print(f"  最终PASS: {total_pass} 条")
        print(f"  最终REJECT: {total_reject} 条")
        print(f"合并结果已保存到: {output_file_path}")
        
        # 保存还原数据的详情（用于审查）
        if restored_count > 0:
            base_name = Path(output_file_path).stem
            output_dir = Path(output_file_path).parent
            restored_detail_file = output_dir / f"{base_name}_restored_details.json"
            
            with open(restored_detail_file, "w", encoding="utf-8") as f:
                json.dump(restored_data, f, ensure_ascii=False, indent=2)
            print(f"还原数据详情已保存到: {restored_detail_file}")
        
        return True
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def merge(base_path, file_name):
    """主函数 - 可以修改这里的文件路径"""
    
    # 设置文件路径 - 请根据实际情况修改
    # base_path = "."
    # file_name = "MatSciInstruct_材料_原始数据_type"
    
    clean_file = os.path.join(base_path, f"{file_name}_clean.json")
    dirty_file = os.path.join(base_path, f"{file_name}_dirty.json")
    output_file = os.path.join(base_path, f"{file_name}_final_merged.json")
    
    print("Move and Merge 工具启动...")
    print(f"Clean文件: {clean_file}")
    print(f"Dirty文件: {dirty_file}")
    print(f"输出文件: {output_file}")
    print("-" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(clean_file):
        print(f"❌ Clean文件不存在: {clean_file}")
        print("请确认文件路径或使用batch_process_directory函数")
        return
    
    if not os.path.exists(dirty_file):
        print(f"❌ Dirty文件不存在: {dirty_file}")
        print("请确认文件路径或使用batch_process_directory函数")
        return
    
    success = process_files(clean_file, dirty_file, output_file)
    
    if success:
        print("\n✅ 处理成功完成！")
    else:
        print("\n❌ 处理失败！")

def batch_process_directory(directory_path):
    """
    批量处理目录中的所有clean/dirty文件对
    
    Args:
        directory_path: 包含clean和dirty文件的目录路径
    """
    import glob
    
    if not os.path.exists(directory_path):
        print(f"错误：目录不存在: {directory_path}")
        return
    
    # 查找所有clean文件
    clean_files = glob.glob(os.path.join(directory_path, "*_clean.json"))
    
    if not clean_files:
        print(f"在目录 {directory_path} 中未找到任何 *_clean.json 文件")
        # 列出目录中的所有json文件以供参考
        all_json_files = glob.glob(os.path.join(directory_path, "*.json"))
        if all_json_files:
            print("目录中现有的json文件:")
            for file in all_json_files[:10]:  # 只显示前10个
                print(f"  - {os.path.basename(file)}")
            if len(all_json_files) > 10:
                print(f"  ... 还有 {len(all_json_files) - 10} 个文件")
        return
    
    print(f"找到 {len(clean_files)} 个clean文件，开始批量处理...")
    
    success_count = 0
    failed_count = 0
    failed_files = []
    
    for clean_file in clean_files:
        # 构造对应的dirty文件路径
        base_name = Path(clean_file).stem.replace("_clean", "")
        dirty_file = os.path.join(directory_path, f"{base_name}_dirty.json")
        output_file = os.path.join(directory_path, f"{base_name}_final_merged.json")
        
        print(f"\n{'='*80}")
        print(f"处理文件对: {base_name}")
        print(f"{'='*80}")
        
        if not os.path.exists(dirty_file):
            print(f"警告：对应的dirty文件不存在: {dirty_file}")
            failed_count += 1
            failed_files.append(base_name)
            continue
        
        success = process_files(clean_file, dirty_file, output_file)
        
        if success:
            success_count += 1
            print(f"✅ {base_name} 处理成功")
        else:
            failed_count += 1
            failed_files.append(base_name)
            print(f"❌ {base_name} 处理失败")
    
    # 最终统计
    print(f"\n{'='*80}")
    print(f"批量处理完成！")
    print(f"{'='*80}")
    print(f"总文件数: {len(clean_files)}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {failed_count}")
    
    if failed_files:
        print(f"\n失败的文件:")
        for file in failed_files:
            print(f"  - {file}")

def process_custom_files(clean_file_path, dirty_file_path, output_file_path=None):
    """
    自定义文件路径处理函数
    
    Args:
        clean_file_path: clean文件完整路径
        dirty_file_path: dirty文件完整路径
        output_file_path: 输出文件路径（可选，默认在clean文件同目录）
    """
    if output_file_path is None:
        base_name = Path(clean_file_path).stem.replace("_clean", "")
        output_dir = Path(clean_file_path).parent
        output_file_path = output_dir / f"{base_name}_final_merged.json"
    
    print("Move and Merge 工具 - 自定义文件处理...")
    print(f"Clean文件: {clean_file_path}")
    print(f"Dirty文件: {dirty_file_path}")
    print(f"输出文件: {output_file_path}")
    print("-" * 60)
    
    success = process_files(clean_file_path, dirty_file_path, output_file_path)
    
    if success:
        print("\n✅ 处理成功完成！")
    else:
        print("\n❌ 处理失败！")
    
    return success

if __name__ == "__main__":
    # 使用方式说明
    print("Move and Merge 工具 v1.0")
    print("="*50)
    print("使用方式:")
    print("1. 单文件处理: 修改main()函数中的路径")
    print("2. 批量处理: 使用batch_process_directory()")
    print("3. 自定义处理: 使用process_custom_files()")
    print("="*50)
    os.chdir(os.path.join(os.path.dirname(__file__), "raw_data"))  # 切换到脚本所在目录的上一级
    # 方式1: 单文件处理模式（需要先确认文件存在）
    base_path = "."
    file_name = "SciKnowEval_processed_openended_filling"
    merge(base_path=base_path, file_name=file_name)
    
    # 方式2: 批量处理模式
    #batch_process_directory("./bio_know_all_data")
    
    # 方式3: 自定义文件处理（示例）
    # process_custom_files(
    #     "./path/to/your_clean.json",
    #     "./path/to/your_dirty.json",
    #     "./path/to/output.json"
    # )