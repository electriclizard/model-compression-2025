import os
import time
import json
import psutil
import shutil
import numpy as np
from tqdm import tqdm

import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoModelForImageClassification,
    ViTImageProcessor,
    logging as hf_logging,
)
import onnxruntime as ort
from optimum.onnxruntime import ORTOptimizer, ORTModelForImageClassification
from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.exporters.onnx import main_export

hf_logging.set_verbosity_warning()


def _ram_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


def preprocess_for_eval(example, feature_extractor, model_is_onnx):
    if model_is_onnx:
        return feature_extractor(example["image"], return_tensors="np")
    else:
        return feature_extractor(example["image"], return_tensors="pt")


def evaluate_performance(model, dataset, feature_extractor, device):
    is_onnx = isinstance(model, ORTModelForImageClassification)

    if not is_onnx:
        model.to(device)
        model.eval()

    latencies, preds, refs = [], [], []
    ram0 = _ram_mb()

    for ex in tqdm(dataset, desc="Evaluation"):
        inputs = preprocess_for_eval(ex, feature_extractor, model_is_onnx=is_onnx)

        if not is_onnx:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        refs.append(ex["labels"])

        tic = time.time()
        with torch.no_grad():
            out = model(**inputs)
        latencies.append(1000 * (time.time() - tic))

        if is_onnx:
            logits = out[0]
        else:
            logits = out.logits.cpu().numpy()

        preds.append(np.argmax(logits, axis=-1).item())

    acc_metric = evaluate.load("accuracy")
    accuracy = acc_metric.compute(predictions=preds, references=refs)["accuracy"]

    if is_onnx:
        model_dir = os.path.dirname(model.model_path)
        size_mb = (
            sum(
                os.path.getsize(os.path.join(model_dir, f))
                for f in os.listdir(model_dir)
            )
            / 2**20
        )
    else:
        temp_dir = "./temp_model_for_size_check"
        os.makedirs(temp_dir, exist_ok=True)
        model_path = os.path.join(temp_dir, "model_weights.pt")
        torch.save(model.state_dict(), model_path)
        size_mb = os.path.getsize(model_path) / 2**20
        shutil.rmtree(temp_dir)

    return {
        "accuracy": accuracy,
        "mean_latency_ms": float(np.mean(latencies)),
        "ram_delta_mb": _ram_mb() - ram0,
        "weights_size_mb": size_mb,
    }


def main():
    MODEL_NAME = "google/vit-base-patch16-224-in21k"
    DATASET_NAME = "beans"
    DEVICE = torch.device("cpu")
    LOCAL_PYTORCH_DIR = "./local_pytorch_model"
    FP32_ONNX_MODEL_DIR = "./fp32_onnx_model"
    OPTIMIZED_ONNX_MODEL_DIR = "./optimized_onnx_model_cpu"

    print(f"Using device: {DEVICE}")

    eval_dataset_raw = load_dataset(DATASET_NAME, split="test")
    feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    labels = eval_dataset_raw.features["labels"].names
    num_labels = len(labels)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    original_model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    baseline_metrics = evaluate_performance(
        model=original_model,
        dataset=eval_dataset_raw,
        feature_extractor=feature_extractor,
        device=DEVICE,
    )
    print(json.dumps(baseline_metrics, indent=2))

    original_model.save_pretrained(LOCAL_PYTORCH_DIR)
    feature_extractor.save_pretrained(LOCAL_PYTORCH_DIR)

    main_export(
        model_name_or_path=LOCAL_PYTORCH_DIR,
        output=FP32_ONNX_MODEL_DIR,
        task="image-classification",
    )

    optimizer = ORTOptimizer.from_pretrained(FP32_ONNX_MODEL_DIR)

    optimization_config = OptimizationConfig(
        optimization_level=99,
        enable_transformers_specific_optimizations=True,
    )

    optimizer.optimize(
        save_dir=OPTIMIZED_ONNX_MODEL_DIR,
        optimization_config=optimization_config,
    )
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    num_physical_cores = psutil.cpu_count(logical=False)
    so.intra_op_num_threads = num_physical_cores
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")

    # Fix affinity error
    os.environ["OMP_NUM_THREADS"] = str(num_physical_cores)
    optimized_model = ORTModelForImageClassification.from_pretrained(
        OPTIMIZED_ONNX_MODEL_DIR, session_options=so
    )

    optimized_metrics = evaluate_performance(
        model=optimized_model,
        dataset=eval_dataset_raw,
        feature_extractor=feature_extractor,
        device=DEVICE,
    )
    print(json.dumps(optimized_metrics, indent=2))

    print("\n\n--- Final Comparison ---")
    print(f"{'Metric':<20} | {'Original (FP32)':<20} | {'Optimized (FP32)':<20}")
    print("-" * 65)
    print(
        f"{'Accuracy':<20} | {baseline_metrics['accuracy']:.4f}{'':<15} | {optimized_metrics['accuracy']:.4f}"
    )
    print(
        f"{'Mean Latency (ms)':<20} | {baseline_metrics['mean_latency_ms']:.2f}{'':<15} | {optimized_metrics['mean_latency_ms']:.2f}"
    )
    print(
        f"{'Weights Size (MB)':<20} | {baseline_metrics['weights_size_mb']:.2f}{'':<15} | {optimized_metrics['weights_size_mb']:.2f}"
    )
    shutil.rmtree(LOCAL_PYTORCH_DIR, ignore_errors=True)
    shutil.rmtree(FP32_ONNX_MODEL_DIR, ignore_errors=True)
    shutil.rmtree(OPTIMIZED_ONNX_MODEL_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
