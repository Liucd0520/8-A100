from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import transformers
from transformers import Trainer, GPTQConfig, deepspeed, BitsAndBytesConfig
import json
import os 
from supervised_dataset import LazySupervisedDataset, SupervisedDataset
from model_save import safe_save_model_for_hf_trainer
import torch

model_name_or_path = '/data/liucd/langchain/Qwen/Qwen1___5-7B-Chat'
data_path = 'data.json'

@dataclass
class ModelArguments:
    model_name_or_path: str = model_name_or_path


@dataclass
class DataArguments:
    data_path: str = field(default=data_path, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False

@dataclass
class LoraArguments:
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",  "up_proj", "gate_proj","down_proj",  
            # "c_attn", "c_proj", "w1", "w2"
        ]  
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    use_lora: bool = True 
    bf16: bool = False
    output_dir: str = 'qwen_output'
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )  # 微调时最大序列长度
    gradient_checkpointing: bool = True 
    report_to: str = 'none'
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta2: float = 0.95
    warmup_ratio: float = 0.01
    lr_scheduler_type: str = 'cosine'
    logging_steps: int = 1
    
    # deepspeed: str = '/data/liucd/BigModel/qwen/Qwen/finetune/ds_config_zero2.json'


args_model = ModelArguments()
args_train = TrainingArguments()
args_lora = LoraArguments()
args_data = DataArguments()

# tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    args_model.model_name_or_path,
    model_max_length=args_train.model_max_length,
    padding_side="right",
    use_fast=False,
)


# model

compute_dtype = (
        torch.float16
        if args_train.fp16
        else (torch.bfloat16 if args_train.bf16 else torch.float32)
    )
config = transformers.AutoConfig.from_pretrained(
        args_model.model_name_or_path,
        cache_dir=args_train.cache_dir,
    )
config.use_cache = False

model = transformers.AutoModelForCausalLM.from_pretrained(
        args_model.model_name_or_path,
        config=config,
        cache_dir=args_train.cache_dir,
        device_map='auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if args_train.use_lora and args_lora.q_lora
        else None,
    )


lora_config = LoraConfig(
            r=args_lora.lora_r,
            lora_alpha=args_lora.lora_alpha,
            target_modules=args_lora.lora_target_modules,
            lora_dropout=args_lora.lora_dropout,
            bias=args_lora.lora_bias,
            task_type="CAUSAL_LM",
        )


if args_lora.q_lora:
     model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args_train.gradient_checkpointing
            )  # 将某些的LN层等从FP16变成FP32


model = get_peft_model(model, peft_config=lora_config) 
model.print_trainable_parameters()

# 调用 model.enable_input_require_grads() 是为了确保在使用 grad_checkpoint 时，模型的输入能够被要求梯度，以便在检查点处能够正确地重新计算梯度。
if args_train.gradient_checkpointing:
    model.enable_input_require_grads()




#  数据集制作
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

# Load Data
data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=args_data, max_len=args_train.model_max_length
    )


# Start trainner
print(args_train)
trainer = Trainer(
    model=model, tokenizer=tokenizer, args=args_train, **data_module
)   
print(trainer)

trainer.train()
trainer.save_state()  # 保存状态

safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args_train.output_dir, bias=args_lora.lora_bias)

