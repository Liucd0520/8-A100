from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


model_path = 'merge_lora'

quant_path = 'awq_qwen'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, model_max_length=512)

# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data='../mit-han-lab___pile-val-backup/', split='validation')

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

