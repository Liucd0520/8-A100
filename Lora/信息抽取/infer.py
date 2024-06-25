from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# # 可选的模型包括: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"

model_name_or_path = 'gptq_qwen'
#model_name_or_path = '/data/liucd/langchain/Qwen/Qwen-7B-Chat'


#tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
     model_name_or_path,
     model_max_length=512,
     padding_side="right",
     use_fast=False,
     trust_remote_code=True,
 )

#tokenizer.pad_token_id = tokenizer.eod_id #  部分tokenizer没有pad_token，例如qwen，将pad_token置为eos_token


model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                             device_map="auto", trust_remote_code=True).eval()

#model.generation_config.top_p=0 # 只选择概率最高的token

# # 可指定不同的生成长度、top_p等相关超参
# model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True)


prompt_template='''
给定一句话：“%s”，请你按步骤要求工作。

步骤1：识别这句话中的城市和日期共2个信息
步骤2：根据城市和日期信息，生成JSON字符串，格式为{"city":城市,"date":日期}

请问，这个JSON字符串是：
'''

Q_list=['2020年4月16号三亚下雨么?','青岛3-15号天气预报','5月6号下雪么, 城市是威海','青岛2023年12月30号有雾霾么?','我打算6月1号去北京旅游，请问天气怎么样？','你们打算1月3号坐哪一趟航班去上海？','小明和小红是8月8号在上海结婚么?',
        '一起去东北看冰雕么, 大概是1月15号左右, 我们3个人一起']

start = time.time()

for Q in Q_list:
    prompt=prompt_template%(Q,)

    A,hist=model.chat(tokenizer,prompt,history=None)
    print('Q:%s\nA:%s\n'%(Q,A))

print(time.time()-start)
