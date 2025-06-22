import os
import time
import json
import psutil
import argparse

import torch
import numpy as np

import evaluate
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME_OR_PATH = "google/vit-base-patch16-224-in21k"
DATASET_NAME = "beans"


def get_ram_usage_mb():
    return psutil.Process().memory_info().rss / 1024**2


def get_vram_usage_mb():
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0


def load_model_and_processor(model_name: str, device: torch.device):
    model = ViTForImageClassification.from_pretrained(MODEL_NAME_OR_PATH).to(device)
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)

    # .pt weights
    if model_name.endswith(".pt"):
        model = torch.load(model_name).to(device)
        # state_dict = torch.load(model_name, map_location=device).state_dict()
        # checkpoint_layers = set(list(state_dict.keys()))
        # # print(checkpoint_layers)
        # original_layers = set(list(model.state_dict().keys()))
        # print(original_layers)
        # # print(original_layers - checkpoint_layers)

        # raise NotImplementedError
        # model.load_state_dict(state_dict)

    model.eval()
    return model, processor


def preprocess_input(example, processor, device: torch.device):
    return processor(example["image"], return_tensors="pt").to(device)


def get_dir_size_mb(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024**2


def evaluate_model(model, dataset, processor, device: torch.device):
    timings = []
    predictions, references = [], []

    start_ram = get_ram_usage_mb()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start_vram = get_vram_usage_mb()

    for example in dataset:
        inputs = preprocess_input(example, processor, device)
        references.append(example["labels"])

        if device.type == "cuda":
            torch.cuda.synchronize()
            start = time.time()  # inference time per sample
            with torch.no_grad():
                output = model(**inputs)
            torch.cuda.synchronize()
        else:
            start = time.time()
            with torch.no_grad():
                output = model(**inputs)
        end = time.time()

        timings.append((end - start) * 1000)
        predictions.append(output.logits.argmax(-1).item())

    end_ram = get_ram_usage_mb()

    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_vram = 0

    return {
        f"{device.type}_time": np.mean(timings),
        "ram_usage": end_ram - start_ram,
        "vram_usage": peak_vram,
        "predictions": predictions,
        "references": references,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = load_dataset(DATASET_NAME, split="test")
    model, processor = load_model_and_processor(
        args.model_name_or_path if args.model_name_or_path else MODEL_NAME_OR_PATH,
        device,
    )
    metric = evaluate.load("accuracy")

    metrics = evaluate_model(model, dataset, processor, device)
    accuracy = metric.compute(
        predictions=metrics["predictions"], references=metrics["references"]
    )
    metrics["accuracy"] = accuracy["accuracy"]

    metrics.pop("predictions")
    metrics.pop("references")

    # for validation
    torch.save(model.state_dict(), "vit_beans/model_weights.pt")
    metrics["weights_size"] = os.path.getsize("vit_beans/model_weights.pt") / (1024**2)

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
