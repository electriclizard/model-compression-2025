from typing import Optional

import torch

from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME_OR_PATH = "google/vit-base-patch16-224-in21k"
CALIBRATION_DATASET_NAME = "beans"


class QuantizationModule:
    def __init__(self, model_name_or_path: str = MODEL_NAME_OR_PATH) -> None:
        self.model_name_or_path = model_name_or_path
        self.processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    def _get_calibration_data(self, batch_size: Optional[int] = 32):
        def preprocess_function(examples):
            images = [image.convert("RGB") for image in examples["image"]]
            inputs = self.processor(
                images=images,
                return_tensors="pt",
            )
            return inputs

        dataset = load_dataset(CALIBRATION_DATASET_NAME, split="train")
        calibration_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=["image"],
        )
        return calibration_dataset

    def _gptq_quantization(self, bits: int, sample_size: Optional[int] = None):
        raise NotImplementedError

    def _awq_quantization(self, bits: int):
        raise NotImplementedError

    def quantize_model(
        self,
        bits: int,
        quantization_method: str = "gptq",
    ):
        quantization_method = quantization_method.lower()

        if isinstance(quantization_method, str):
            if quantization_method == "dynamic_quantization":
                if bits in [8, 16]:
                    if bits == 8:
                        dtype_after_quantization = torch.qint8
                    else:
                        dtype_after_quantization = torch.float16

                    dynamic_model = ViTForImageClassification.from_pretrained(
                        self.model_name_or_path, device="auto"
                    )
                    model_quantized = torch.ao.quantization.quantize_dynamic(
                        dynamic_model,
                        qconfig_spec={torch.nn.Linear},
                        dtype=dtype_after_quantization,
                    )

            elif quantization_method == "gptq":
                model_quantized = self._gptq_quantization(bits, sample_size=32)
            elif quantization_method == "awq":
                model_quantized = self._awq_quantization(bits)

        print()
        print(
            f"Current quantization is {bits}, and have following method: {quantization_method}:"
        )
        print()
        torch.save(
            model_quantized,
            f"/home/astikhono7/projects/model-compression-2025/models/vit-base-patch16-224-in21k-bits-{bits}_{quantization_method}_wo_pruning.pt",
        )

        return model_quantized


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_module = QuantizationModule(MODEL_NAME_OR_PATH)

    quint8_model = quant_module.quantize_model(
        bits=8, quantization_method="dynamic_quantization"
    )
    bf16_model = quant_module.quantize_model(
        bits=16, quantization_method="dynamic_quantization"
    )

    print("Successfully saved.")
