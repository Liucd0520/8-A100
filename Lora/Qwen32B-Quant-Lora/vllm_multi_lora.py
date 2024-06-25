from openai import OpenAI
import time
from typing import List
from pydantic import BaseModel


# 非量化版本: 7B 模型
"""
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen1.5-7B-Chat \
      --model /data/liucd/langchain/Qwen/Qwen1___5-7B-Chat  --port 1122 \
     --enable-lora  --max-lora-rank 64 \
      --lora-modules weather-lora=/data/liucd/Lora/对话_Qwen1.5/peft_qwen1.5/qwen_output_weather  huanhuan-lora=/data/liucd/Lora/对话_Qwen1.5/peft_qwen1.5/qwen_output_huanhuan
"""


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:1122/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

prompt_weather='''
给定一句话：“2020年4月16号三亚下雨么?”，请你按步骤要求工作。

步骤1：识别这句话中的城市和日期共2个信息
步骤2：根据城市和日期信息，生成JSON字符串，格式为{"city":城市,"date":日期}

请问，这个JSON字符串是：
''' 

prompt_huanhuan = '你是谁'


temp_jianchasuojian = '左肺上、下叶见多发团片状、斑点、结节高密度影，边界不清，左肺上叶为著伴钙化，周围见多发类结节影，邻近胸膜粘连增厚；左肺上叶体积缩小；左侧胸腔内可见积气、积液及引流管影；另两肺见多发小结节影，直径约3-4mm，界清；余两肺纹理略增多，未见明显异常密度占位影；两肺门结构尚清，纵隔居中，其内未见明显肿大淋巴结；右侧胸腔、心包腔内未见明显积液影；心影无殊.'
temp_jianchazhenduan = '对比2023-03-29片： 1.左肺多发多形性病变，左肺上叶为著，左侧胸膜增厚，VP-RADS 2类，结核首先考虑，请结合临床实验室检查及治疗后复查。 2.左侧液气胸胸腔闭式引流术后改变，积气较前略吸收。 3.两肺多发小结节，LUNG-RADs:2类，大致相仿，建议年度复查。 附见：肝内致密影。'

prompt_base = f'你是一位资深医学专家，目前需要从报告\'检查所见\'中的关于肺结节的核心字段，主要包括\'原文句子\',\'结节部位\',\'结节大小\',\'结节类型\',\'结节边界\', \'淋巴>结情况\', 需要从\'检查诊断\'中的信息用于提取“LUNG-RADs”字段，然后输出一个json格便于后续的处理, 返回形式类似 list of dict 格式:\n\n#要求：\n1、请输出一个合法的 JSON 字符串, 参考 sample output 的输出格式;\n2、“检查所见”字段信息需要抽取对应原文，找到肺结节相关信息;\n3、“检查所见”字段信息不需要抽取原文，需要协助找到"LUNG-RADs"的值，只有“结节类>型”为肺结节类型时，才能有值，否则为空"-";\n\n\n#example:\n#sample input：\n检查所见: 左肺上、下叶见多发团片状、斑点、结节高密度影，边界不清，左肺上叶为著伴钙化，周围见多发类结节影，邻近胸膜粘连增厚；左肺上叶体积缩小；左侧胸腔内可见积气、积液及引流管影；另两肺见多发小结节影，直径约3-4mm，界清；余两肺纹理略增多，未见明显异常密度占位影；两肺门结构尚清，纵隔居中，其内未见明显肿大淋巴结；右侧胸腔、心包腔内未见明显积液影；心影无殊.\n\n检查诊断: 对比2023-03-29片： 1.左肺多发多形性病变，左肺上叶为著，左侧胸膜增厚，VP-RADS 2类，结核首先考虑，请结合临床实验室检查及治疗后复查。 2.左侧液气胸胸腔闭式引流术后改变，积气较前略吸收。 3.两肺多发小结节，LUNG-RADs:2类，大致相仿，建议年度复查。 >附见：肝内致密影。\n\n#sample output:\n[\n    {{\n        \'原文句子\': \'左肺上、下叶见多发团片状、斑点、结节高密度影，边界不清，左肺上叶为著伴钙化，周围见多发类结节影，>邻近胸膜粘连增厚\',\n        \'结节部位\': \'左肺上、下叶\',\n        \'结节大小\': \'-\',\n        \'结节类型\': \'团片状、斑点、结节高密度影,钙化\',\n        \'结节边界\': \'边界不清\',\n        \'淋巴结情况\': \'-\',\n        \'LUNG-RADs\': \'2类\',\n    }},\n    {{\n        \'原文句子\': \'另两肺见多发小结节影，直径约3-4mm，界清\',\n        \'结节部位\': \'两肺\',\n        \'结节大小\': \'直径约3-4mm\',\n        \'结节类型\': \'小结节影\',\n        \'结节边界\': \'界清\',\n        \'淋巴结情况\': \'-\',\n        \'LUNG-RADs\': \'2类\',\n    }},\n]\n\n#input:\n检查所见:{temp_jianchasuojian}\n\n检查诊断:{temp_jianchazhenduan}\n\n#output:\n\n'

class LungNode(BaseModel):
    部位: List[str]
    大小: List[str] 
    
json_template = LungNode.schema()


# base 调用
chat_response = client.chat.completions.create(
    model='Qwen1.5-7B-Chat',
    messages=[
        {"role": "user", "content": prompt_base},
    ],

    extra_body={
        "guided_json":json_template,
        "response_format":{'type': 'json_object'}
    }
)

print(chat_response.choices[0].message.content)

# 天气调用
chat_response = client.chat.completions.create(
    model = 'weather-lora',
    messages=[
        
        {"role": "system", "content": "你是一个人工智能助手"},
        {"role": "user", "content": prompt_weather},
    ]
)
print(chat_response.choices[0].message.content)

# huanhuan
chat_response = client.chat.completions.create(
    
    model = 'huanhuan-lora',
    messages=[
        
        {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
        {"role": "user", "content": prompt_huanhuan},
    ]
)
print(chat_response.choices[0].message.content)


