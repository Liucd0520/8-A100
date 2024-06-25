from vllm import LLM, SamplingParams
import time


# Sample prompts.
prompts = [
    "告诉我有关北京的特产",
] * 5

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2000)

# Create an LLM.
llm = LLM(model='gptq_qwen/',quantization='gptq',
         trust_remote_code=True)

prompt_template='''
给定一句话：“%s”，请你按步骤要求工作。

步骤1：识别这句话中的城市和日期共2个信息
步骤2：根据城市和日期信息，生成JSON字符串，格式为{"city":城市,"date":日期}

请问，这个JSON字符串是：
'''

Q_list=['2020年4月16号三亚下雨么?','青岛3-15号天气预报','5月6号下雪么, 城市是威海','青岛2023年12月30号有雾霾么?','我打算6月1号去北京旅游，请问天气怎么样？','你们打算1月3号坐哪一趟航班去上海？','小明和小红是8月8号在上海结婚么?',
        '一起去东北看冰雕么, 大概是1月15号左右, 我们3个人一起']

prompts = [prompt_template%(Q,) for Q in Q_list]

start = time.time()
outputs = llm.generate(prompts, sampling_params)
print(outputs)
print(time.time()-start)

