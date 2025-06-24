import os
import time
from rich.table import Table

import psutil
import torch
import numpy as np


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_usage():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 * 1024)


def measure_performance(model, test_dataset, device, model_type='original', 
                      centroids=None, indices=None, non_clustered=None):
    print(f'Measuring performance for {model_type} model on {device}...')

    model = model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0
    inference_times = []

    ram_usage = get_memory_usage()
    vram_usage = get_gpu_memory_usage() if device == 'cuda' else 0

    # Warmup
    first_sample = test_dataset[0]
    with torch.no_grad():
        for _ in range(10):
            _ = model(first_sample['pixel_values'].unsqueeze(0).to(device))

    # Calculate real model size based on compression
    if model_type in ['clustered', 'fine-tuned clustered'] and centroids and indices:
        # Calculate compressed size
        compressed_size = 0
        for name in centroids:
            compressed_size += centroids[name].nbytes  # centroids stored as float16
            compressed_size += indices[name].nbytes    # indices stored as uint16
        
        if non_clustered:
            for name in non_clustered:
                compressed_size += non_clustered[name].nbytes  # non-clustered params as float16
        
        model_size_mb = compressed_size / (1024 * 1024)
    else:
        # Original model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # Actual inference testing
    for i in range(len(test_dataset)):
        try:
            sample = test_dataset[i]
            pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
            label = sample['labels']

            start_time = time.time()
            with torch.no_grad():
                outputs = model(pixel_values)
            if device == 'cuda':
                torch.cuda.synchronize()
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)

            predictions = outputs.logits.argmax(-1).cpu().item()
            total_correct += int(predictions == label)
            total_samples += 1

        except Exception as e:
            print(f'Error processing sample {i}: {str(e)}')
            continue

    # Calculate metrics
    avg_time_ms = np.mean(inference_times) if inference_times else 0
    accuracy = (total_correct / total_samples) * 100 if total_samples else 0

    return {
        'Model': model_type,
        'Size (MB)': f'{model_size_mb:.2f}',
        'Avg Inference (ms)': f'{avg_time_ms:.2f}',
        'RAM (MB)': f'{ram_usage:.2f}',
        'VRAM (MB)': f'{vram_usage:.2f}',
        'Accuracy (%)': f'{accuracy:.2f}',
        'Samples Tested': total_samples
    }