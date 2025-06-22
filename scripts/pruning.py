from typing import Optional

import torch
from torch.nn.utils import prune

from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME_OR_PATH = "google/vit-base-patch16-224-in21k"
CALIBRATION_DATASET_NAME = "beans"


class PruningModule:
    def __init__(self, model_name_or_path: str = MODEL_NAME_OR_PATH) -> None:
        self.model_name_or_path = model_name_or_path
        self.processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    def _get_calibration_data(self, batch_size: Optional[int] = 32):
        def preprocess_function(examples):
            images = [image.convert("RGB") for image in examples["image"]]
            data_inputs = self.processor(
                images=images,
                return_tensors="pt",
            )
            return data_inputs

        dataset = load_dataset(CALIBRATION_DATASET_NAME, split="train")
        calibration_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=["image"],
        )
        return calibration_dataset

    def _unstructured_pruning(self, fraction_params_to_prune: float = 0.5):
        modules_to_prune = []
        model = ViTForImageClassification.from_pretrained(
            MODEL_NAME_OR_PATH, device_map="auto"
        )

        for module_name, torch_module in model.named_modules():
            if (
                isinstance(torch_module, torch.nn.Linear)
                and module_name != "classifier"
            ):
                modules_to_prune.append((torch_module, "weight"))

        prune.global_unstructured(
            parameters=modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=fraction_params_to_prune,
        )

        return model

    def _structured_pruning(self, fraction_params_to_prune: float = 0.5):
        model = ViTForImageClassification.from_pretrained(
            MODEL_NAME_OR_PATH, device_map="auto"
        )

        for module_name, torch_module in model.named_modules():
            if (
                isinstance(torch_module, torch.nn.Linear)
                and module_name != "classifier"
            ):
                torch_module = prune.random_structured(
                    torch_module,
                    "weight",
                    dim=1,
                    amount=fraction_params_to_prune,
                )
                assert int(sum(torch.sum(torch_module.weight, dim=0) == 0)) != 0

        return model

    def prune_model(
        self,
        prune_method: str = "structured",
    ):
        prune_method = prune_method.lower()
        if isinstance(prune_method, str):
            if prune_method == "unstructured":
                pruned_model = self._unstructured_pruning()
            elif prune_method == "structured":
                pruned_model = self._structured_pruning()
            else:
                raise NotImplementedError

        print()
        print(f"Current method: {prune_method}:")
        print(pruned_model)
        print()
        torch.save(
            pruned_model,
            f"/home/astikhono7/projects/model-compression-2025/models/vit-base-patch16-224-in21k-{prune_method}_pruning.pt",
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prune_module = PruningModule(MODEL_NAME_OR_PATH)

    unstructured_pruning = prune_module.prune_model(prune_method="unstructured")
    structured_pruning = prune_module.prune_model(prune_method="structured")

    print("Successfully saved.")
