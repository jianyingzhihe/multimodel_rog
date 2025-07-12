from modelscope import AutoModelForCausalLM, AutoTokenizer


# 模型路径
model_path = "/root/autodl-tmp/RoG/qwen/output/v11-20250710-145738/checkpoint-201"

# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# 系统提示词
with open("../remp/story.txt", "r", encoding="utf-8") as f:
    sys_prompt = f.read()

# 对话循环
while True:
    try:
        quest = input("User: ").strip()
        if not quest:
            break

        # 构建 chat 消息结构
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": quest}
        ]

        # 使用 Qwen 的 chat 模板构造输入
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # ✅ 注意：这里不要加 [] 把 text 变成列表，除非你要批量处理
        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # 生成回复
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )

        # 去除输入部分，只保留生成内容
        generated_ids = generated_ids[:, model_inputs['input_ids'].shape[-1]:]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Assistant: {response}")

    except KeyboardInterrupt:
        print("\n👋 推理已手动中断。")
        break