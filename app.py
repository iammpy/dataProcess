import gradio as gr
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from model import call_huoshan, call_openai ,OpenaiTranslator
# # --- 为了可独立运行，我们先模拟你的调用函数 ---
# def call_huoshan(prompt: str, model: str):
#     print(f"后台：开始调用火山模型 {model}...")
#     if "doubao" in model:
#         time.sleep(3)
#     elif "qwen" in model:
#         # 模拟这个模型会卡住很长时间
#         print(f"后台：qwen 模型开始了一个超长任务...")
#         time.sleep(60) # 这个任务会超过我们的30秒超时设置
#     else: # r1
#         time.sleep(5)
#     response = f"这是【{model}】对 '{prompt}' 的回答。"
#     print(f"后台：{model} 调用完成。")
#     return response

# def call_openai(prompt: str):
#     print("后台：开始调用 OpenAI...")
#     time.sleep(4)
#     response = f"这是【OpenAI】对 '{prompt}' 的回答。"
#     print("后台：OpenAI 调用完成。")
#     return response
# # --- 模拟函数结束 ---


# --- 这是新的、带超时的生成器函数 ---
def stream_all_models_with_timeout(system_prompt, user_prompt):
    if not user_prompt:
        yield "Prompt不能为空", "Prompt不能为空", "Prompt不能为空", "Prompt不能为空","Prompt不能为空"
        return
    
    prompt = f"{system_prompt}\n\n用户问题：\n{user_prompt}"
    # print(f"后台：收到的最终Prompt是：\n{prompt}")

    # 1. 初始反馈
    initial_feedback = "正在思考中..."
    yield initial_feedback, initial_feedback, initial_feedback, initial_feedback,initial_feedback

    models_to_call = {
        0: ("huoshan", "doubao"),
        1: ("huoshan", "qwen"), # 这个会超时
        2: ("huoshan", "r1"),
        3: ("openai", None),
        4: ("OpenaiTranslator", None)  # 新增 OpenaiTranslator 模型
    }
    
    results = [initial_feedback] * len(models_to_call)
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_index = {}
        for index, (func_name, model_name) in models_to_call.items():
            if func_name == "huoshan":
                future = executor.submit(call_huoshan, prompt, model_name)
            elif func_name == "openai":
                future = executor.submit(call_openai, prompt)
            elif func_name == "OpenaiTranslator":
                future = executor.submit(OpenaiTranslator, user_prompt)
            future_to_index[future] = index

        # 2. 核心修改：为 as_completed 循环中的 result() 添加超时
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            model_info = models_to_call[index]
            try:
                # 设置30秒超时！
                _, result = future.result(timeout=300)
                results[index] = result
            except TimeoutError:
                error_message = f"模型 {model_info} 调用超时（超过300秒）！"
                print(f"后台：{error_message}")
                results[index] = error_message
            except Exception as e:
                error_message = f"模型 {model_info} 调用失败: {e}"
                print(f"后台：{error_message}")
                results[index] = error_message
            
            # 3. 实时更新UI
            yield tuple(results)

    print("后台：所有任务处理完毕（包括已超时的）。")


# --- 创建 Gradio Blocks ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 一键调用多模型工具")
    # gr.Markdown("内置30秒超时和取消功能，防止应用卡死。")

    # --- 在这里增加 ---
    # 1. 创建 System Prompt 输入框，并为其设置一个默认值
    system_prompt_input = gr.Textbox(
        label="System Prompt (系统提示词)",
        value="对于下面的问题，模型拒绝或者接受的原因合理吗，给出你的判断。用中文回答。", # 这是一个不错的默认值
        lines=3,
        interactive=True # 确保它是可编辑的
    )
    # --- 增加结束 ---
    
    with gr.Row():
        prompt_input = gr.Textbox(lines=5, placeholder="在这里输入Prompt...", label="统一输入", scale=3)
        with gr.Column(scale=1):
            submit_button = gr.Button("提交调用", variant="primary")
            cancel_button = gr.Button("取消执行") # 新增取消按钮
    with gr.Row():
        output_e=gr.Textbox("翻译结果", interactive=True)
    with gr.Row():
        with gr.Accordion("doubao", open=True):  # open=True 让它默认保持展开
            output_a = gr.Markdown()
        with gr.Accordion("qwen", open=True):  # open=True 让它默认保持展开
            output_b = gr.Markdown()
    with gr.Row():
        with gr.Accordion("r1", open=True):  # open=True 让它默认保持展开
            output_c = gr.Markdown()
        with gr.Accordion("openai", open=True):  # open=True 让它默认保持展开
            output_d = gr.Markdown()

    # 将所有输出组件放入一个列表，方便管理
    all_outputs = [output_a, output_b, output_c, output_d,output_e]

    # 提交按钮的点击事件
    submit_event = submit_button.click(
        fn=lambda: ("", "", "", ""), # 点击后立刻清空输出
        outputs=all_outputs,
        queue=False # 这一步很快，不需要排队
    ).then(
        fn=stream_all_models_with_timeout, # 然后开始真正的任务
        inputs=[system_prompt_input,prompt_input],
        outputs=all_outputs
    )

    # 取消按钮的点击事件
    # `cancels` 参数是关键，它指向需要被取消的事件
    cancel_button.click(
        fn=None, # 取消按钮本身不需要执行函数
        inputs=None,
        outputs=None,
        cancels=[submit_event] # 指定要取消的事件
    )

# 启动服务
# demo.queue(max_size=10)
demo.queue(default_concurrency_limit=10)
demo.launch()