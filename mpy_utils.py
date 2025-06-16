
import os
import torch
from mp_api.client import MPRester
import matgl
from matgl.ext.pymatgen import Structure2Graph
from  matgl.ext.ase import Relaxer # 导入专门用于结构弛豫的工具

# ==============================================================================
# 步骤 1: 一次性加载所有需要的模型
# ==============================================================================
print("--- 正在加载所有必需的 MATGL 模型 ---")

pes_model = matgl.load_model("CHGNet-MPtrj-2024.2.13-11M-PES")
print("势能面 (PES) 模型加载成功！")

model_wrapper = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
print("带隙 (BandGap) 模型加载成功！")



# ==============================================================================
# 步骤 2: 定义一个可重复使用的、稳健的预测函数
# ==============================================================================
def predict_property_for_modified_structure(
    initial_structure,
    modification_type: str,
    modification_details: dict,
    potential_model, 
    property_model
    
):
    """
    一个通用的函数，用于修改结构、进行弛豫，并预测最终性质。

    Args:
        initial_structure: 初始的 pymatgen Structure 对象。
        modification_type: 修改类型，如 "substitute", "remove", "add"。
        modification_details: 一个字典，包含修改的细节。
            - for "substitute": {"from": "Fe", "to": "Al"}
            - for "remove": {"element": "Fe"}
            - for "add": {"element": "Li", "coords": [0.5, 0.5, 0.5]}
        potential_model: 用于结构弛豫的 matgl PES 模型。
        property_model: 用于最终性质预测的 matgl 模型。
    """
    # print(f"\n--- 开始处理新任务：对 {initial_structure.composition.reduced_formula} 进行 {modification_type} 操作 ---")
    
    # --- 2.1. 根据指令修改结构 ---
    # print("步骤 2.1: 正在应用原子修改...")
    modified_structure = initial_structure.copy()
    
    if modification_type == "substitute":
        from_el = modification_details["from"]
        to_el = modification_details["to"]
        modified_structure.replace_species({from_el: to_el})
        # print(f"已将 {from_el} 替换为 {to_el}。新化学式: {modified_structure.composition.reduced_formula}")

    # --- 新增的 'remove' 逻辑 ---
    elif modification_type == "remove":
        element_to_remove = modification_details["element"]
        # 找到所有待删除元素在结构中的索引
        indices_to_remove = [i for i, site in enumerate(modified_structure) if site.species_string == element_to_remove]
        
        if not indices_to_remove:
            print(f"警告: 在结构中未找到要删除的元素 {element_to_remove}。")
        else:
            # 从结构中删除这些位置上的原子
            modified_structure.remove_sites(indices_to_remove)
            print(f"已移除所有 {element_to_remove} 原子。新化学式: {modified_structure.composition.reduced_formula}")

    # --- 新增的 'add' 逻辑 ---
    elif modification_type == "add":
        element_to_add = modification_details["element"]
        coords_to_add = modification_details["coords"]
        # 在指定坐标添加新原子
        # 假设 coords_are_cartesian=False，即我们提供的是分数坐标
        modified_structure.append(species=element_to_add, coords=coords_to_add, coords_are_cartesian=False)
        print(f"已在分数坐标 {coords_to_add} 添加 {element_to_add} 原子。新化学式: {modified_structure.composition.reduced_formula}")
        
    else:
        raise ValueError(f"不支持的修改类型: {modification_type}")

    # --- 2.2. 对新结构进行弛豫，找到稳定构型 ---
    print("步骤 2.2: 正在使用 PES 模型进行结构弛豫...")
    relaxer = Relaxer(potential=potential_model)
    relaxation_results = relaxer.relax(modified_structure, fmax=0.1)

    final_energy = relaxation_results["trajectory"].energies[-1]
    
    relaxed_structure = relaxation_results["final_structure"]
    print(f"结构弛豫完成。最终能量: {float(final_energy):.3f} eV")

    # --- 2.3. 使用弛豫后的结构预测最终性质 ---
    print("步骤 2.3: 正在对弛豫后的结构预测带隙...")
    # 对于这个多保真度模型，我们使用 state_attr=0 来预测 PBE 带隙，
    # 以便和 Materials Project 数据库中的默认带隙值进行对比。
    state_attr = torch.tensor([3])
    
    # 直接调用高级接口，所有的数据转换和修正都在后台自动完成
    prediction_tensor = property_model.predict_structure(
        structure=relaxed_structure,
        state_attr=state_attr
    )
    
    return prediction_tensor.item()

def predict_bandgap_for_structure(
    formula: str,
    parsed_json: str,
):
    # --- 3.1 获取一个初始结构作为示例 ---
    API_KEY = "bSpFQg1jCJpFG4CARe0NiSUyXKke56OF" # 已使用您提供的密钥
    initial_structure = None
    # formula= "LiFePO4F"  # 目标化学式
    with MPRester(api_key=API_KEY) as mpr:
        # 以 LiFePO4F (mp-755813) 为例
        try:
            docs = mpr.materials.search(formula=formula, fields=["structure"])
            if docs:
                initial_structure = docs[0].structure
        except Exception as e:
            print(f"获取化学式 {formula} 的结构时出错: {str(e)}")
            return e
    # parsed_json = {
    #     "modification_type": "substitute",
    #     "new_material_formula": "LiAlPO4F", # 我们可以在最后打印时使用
    #     "details": {
    #         "from_element": "Fe",
    #         "to_element": "Al"
    #     }
    # }
    if initial_structure:
        modification_details_for_function = {
            "from": parsed_json["details"].get("from_element"),
            "to": parsed_json["details"].get("to_element"),
            "element": parsed_json["details"].get("element"),
            "coords": parsed_json["details"].get("coords")
        }

        # 这就是您在并发任务中需要调用的核心部分
        predicted_bandgap = predict_property_for_modified_structure(
            initial_structure=initial_structure,
            modification_type=parsed_json["modification_type"], # 使用 'modification_type' 键
            modification_details=modification_details_for_function, # 使用转换后的details字典
            potential_model=pes_model,
            property_model=model_wrapper
        )
        
        # 在打印最终结果时，我们可以用上JSON里的 'new_material_formula'
        final_formula = parsed_json.get("new_material_formula", "N/A")

        return predicted_bandgap, final_formula



if __name__ == "__main__":
    # 示例调用
    formula = "BaCaCeSnO6"  # 目标化学式
    parsed_json = {
        "modification_type": "substitute",
        "new_material_formula": "LiAlPO4F",  # 我们可以在最后打印时使用
        "details": {
            "from_element": "Fe",
            "to_element": "Al"
        }
    }
    
    bandgap, final_formula = predict_bandgap_for_structure(formula, parsed_json)
    print(f"预测的带隙: {bandgap:.3f} eV, 新材料化学式: {final_formula}")