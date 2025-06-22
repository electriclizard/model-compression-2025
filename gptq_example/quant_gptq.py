import os
import json

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig
from optimum.gptq import GPTQQuantizer


model_id = "IlyaGusev/saiga_llama3_8b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

dataset_id = "wikitext2"

quantizer = GPTQQuantizer(
        bits=8,
        dataset=dataset_id,
        model_seqlen=4096
        )

quantized_model = quantizer.quantize_model(model, tokenizer)

# save_folder = "/models/llama3-vllm/1/weights-8bit"
save_folder = "/src/quant_llama"
quantized_model.save_pretrained(save_folder, safe_serialization=True)

# load fresh, fast tokenizer and save it to disk
tokenizer = AutoTokenizer.from_pretrained(model_id).save_pretrained(save_folder)

# save quantize_config.json for TGI
with open(os.path.join(save_folder, "quantize_config.json"), "w", encoding="utf-8") as f:
    quantizer.disable_exllama = False
    json.dump(quantizer.to_dict(), f, indent=2)

# For inference and production load we want to leverage the exllama kernels. Therefore we need to change the config.json
with open(os.path.join(save_folder, "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)
    config["quantization_config"]["disable_exllama"] = False
    with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
