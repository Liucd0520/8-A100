from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM


save_path = 'merge_lora'

path_to_adapter = 'qwen_output/'
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()



merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary. 
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(save_path, max_shard_size="2048MB", safe_serialization=True)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
    trust_remote_code=True
)
tokenizer.save_pretrained(save_path)

