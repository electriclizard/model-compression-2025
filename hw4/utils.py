import os
import time

import psutil
import torch


def get_memory_usage():
    """Get RAM usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_usage():
    """Get VRAM usage in MB"""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 * 1024)


def measure_performance(model, test_dataset, device, model_type="original"):
    """Measure model performance on full test dataset"""
    print(f"Measuring performance for {model_type} model on {device}...")

    model = model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0
    inference_times = []

    ram_usage = get_memory_usage()
    vram_usage = get_gpu_memory_usage() if device == "cuda" else 0

    first_sample = test_dataset[0]
    with torch.no_grad():
        for _ in range(10):
            _ = model(first_sample["pixel_values"].unsqueeze(0).to(device))

    for i in range(len(test_dataset)):
        try:
            sample = test_dataset[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            label = sample["labels"]

            start_time = time.time()
            with torch.no_grad():
                outputs = model(pixel_values)
            if device == "cuda":
                torch.cuda.synchronize()
            inference_times.append((time.time() - start_time) * 1000)  # ms

            predictions = outputs.logits.argmax(-1).cpu().item()
            total_correct += int(predictions == label)
            total_samples += 1

        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            continue

    avg_time_ms = sum(inference_times) / len(inference_times) if inference_times else 0
    accuracy = (total_correct / total_samples) * 100 if total_samples else 0
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    return {
        "Model": model_type,
        "Size (MB)": f"{model_size_mb:.2f}",
        "Avg Inference (ms)": f"{avg_time_ms:.2f}",
        "RAM (MB)": f"{ram_usage:.2f}",
        "VRAM (MB)": f"{vram_usage:.2f}",
        "Accuracy (%)": f"{accuracy:.2f}",
        "Samples Tested": total_samples
    }