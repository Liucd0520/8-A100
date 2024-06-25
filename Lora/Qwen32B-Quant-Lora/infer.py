from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda:0" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "merge_lora_weather",
    torch_dtype="auto",
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("merge_lora_weather")

prompt_template='''
给定一句话：“%s”，请你按步骤要求工作。

步骤1：识别这句话中的城市和日期共2个信息
步骤2：根据城市和日期信息，生成JSON字符串，格式为{"city":城市,"date":日期}

请问，这个JSON字符串是：
'''
Q_list=['2020年4月16号三亚下雨么?','青岛3-15号天气预报','5月6号下雪么, 城市是威海','青岛2023年12月30号有雾霾么?','我打算6月1号去北京旅游，请问天气怎么样？','你们打算1月3号坐哪一趟航班去上海？','小明和小红是8月8号在上海结婚么?',
        '一起去东北看冰雕么, 大概是1月15号左右, 我们3个人一起']

import time 
start = time.time()
for Q in Q_list:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_template%(Q)}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) 

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
print(time.time() - start)