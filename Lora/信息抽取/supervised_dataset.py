from torch.utils.data import Dataset
import transformers
from typing import Dict
import torch 
from transformers.trainer_pt_utils import LabelSmoother
import json


IGNORE_TOKEN_ID = LabelSmoother.ignore_index  # ，通常会将某些特定的标签视为不需要进行平滑处理的标签，
                # 例如填充标签（padding tokens）或者特殊标记（如开始标记、结束标记等）。
                # 将原始的独热编码标签转换为软性标签。如独热编码标签 [0, 1, 0] 转换为软性标签 [0.1, 0.8, 0.1]。

# 采用了延迟加载的方式，数据在需要时才进行预处理和加载，适合处理大规模数据集

TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\n 你是一个人工智能助手。<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["messages"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["target_ids"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret



# 一次性加载和预处理所有数据，适合处理较小规模的数据集，选择哪种加载方式取决于数据集的大小和内存限制

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(SupervisedDataset, self).__init__()
    

        messages = [example["messages"] for example in raw_data]
        data_dict = preprocess(messages, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.target_ids = data_dict["target_ids"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.target_ids[i],
            attention_mask=self.attention_mask[i],
        )





def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                chat_template=TEMPLATE,
                tokenize=True,
                add_generation_prompt=False,
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
        )
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )



if __name__ == '__main__':

    data_path = 'data.json'
    train_json = json.load(open(data_path, 'r'))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        '/data/liucd/langchain/Qwen/Qwen1___5-7B-Chat',
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    SupervisedDataset(raw_data=train_json, tokenizer=tokenizer, max_len=8000)

