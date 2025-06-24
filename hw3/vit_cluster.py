import os
import time
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.datasets import CIFAR10
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from sklearn.cluster import KMeans
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from torch.utils.data import random_split
from dataset import CIFAR10Dataset
from utils import measure_performance
import faiss


console = Console()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_model_info(model, name):
    table = Table(title=f'{name} Model Info')
    table.add_column('Parameter', style='cyan')
    table.add_column('Value', style='magenta')
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    table.add_row('Total parameters', f'{total_params:,}')
    table.add_row('Trainable parameters', f'{trainable_params:,}')
    table.add_row('Device', next(model.parameters()).device.type)
    
    console.print(table)

def train_model(train_dataset, eval_dataset, processor, device, epochs=5):
    console.rule('[bold blue]Training ViT Model[/bold blue]')
    
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=10,
        ignore_mismatched_sizes=True
    ).to(device)
    
    print_model_info(model, 'Original ViT')
    
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        eval_strategy='epoch',
        save_strategy='epoch',
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        remove_unused_columns=False,
        report_to='none',
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    console.print(f'[bold]Training for {epochs} epochs[/bold]')
    trainer.train()
    
    model_dir = './trained_model/'
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    console.print(f'[bold green]Model saved to [cyan]{model_dir}[/cyan][/bold green]')
    
    return model

def save_compressed_model(model, centroids, indices, save_dir):
    """Сохранение модели в сжатом формате (центроиды + индексы)"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.config.save_pretrained(save_dir)
    
    compressed_data = {
        'centroids': {k: v.astype(np.float16) for k, v in centroids.items()},  # Сжатие центроидов
        'indices': {k: v.astype(np.uint16) for k, v in indices.items()},      # Индексы как uint16
        'non_clustered': {}
    }
    
    for name, param in model.named_parameters():
        if name not in indices:
            compressed_data['non_clustered'][name] = param.data.cpu().numpy().astype(np.float16)
    
    torch.save(compressed_data, os.path.join(save_dir, 'compressed_weights.pt'))

def load_compressed_model(save_dir, device):
    """Загрузка сжатой модели"""
    config = ViTForImageClassification.config_class.from_pretrained(save_dir)
    model = ViTForImageClassification(config).to(device)
    
    compressed_data = torch.load(os.path.join(save_dir, 'compressed_weights.pt'), map_location='cpu')
    
    for name, param in model.named_parameters():
        if name in compressed_data['indices']:
            indices = compressed_data['indices'][name]
            centroids = compressed_data['centroids'][name]
            clustered_weights = centroids[indices].reshape(param.shape)
            param.data = torch.from_numpy(clustered_weights).float().to(device)
        elif name in compressed_data['non_clustered']:
            param.data = torch.from_numpy(compressed_data['non_clustered'][name]).float().to(device)
    
    return model

def get_compressed_size(centroids, indices, non_clustered):
    total_bytes = 0
    
    for v in centroids.values():
        total_bytes += v.nbytes
    
    for v in indices.values():
        total_bytes += v.nbytes
    
    for v in non_clustered.values():
        total_bytes += v.nbytes
    
    return total_bytes / (1024 * 1024)

def cluster_weights(model, n_clusters=16):
    clustered_model = model.__class__(model.config).to(device)
    clustered_model.load_state_dict(model.state_dict())
    
    centroids_dict = {}
    indices_dict = {}
    non_clustered_params = {}
    
    for name, param in clustered_model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            console.print(f'[yellow]Clustering layer: {name}[/yellow]')
            
            weights = param.data.cpu().numpy()
            original_shape = weights.shape
            weights_flat = weights.reshape(-1, 1).astype('float32')
            
            kmeans = faiss.Kmeans(d=1, k=n_clusters, gpu=False)
            kmeans.train(weights_flat)
            
            _, labels = kmeans.index.search(weights_flat, 1)
            centroids = kmeans.centroids.astype(np.float16)  # Сжатие центроидов
            
            centroids_dict[name] = centroids.flatten()
            indices_dict[name] = labels.reshape(original_shape).astype(np.uint16)  # Сжатие индексов
            
            clustered_weights = centroids[labels].reshape(original_shape)
            param.data = torch.from_numpy(clustered_weights).to(param.device).float()
        else:
            non_clustered_params[name] = param.data.cpu().numpy().astype(np.float16)
    
    return clustered_model, centroids_dict, indices_dict, non_clustered_params


def fine_tune_clustered_model(model, train_dataset, eval_dataset, epochs=3):
    console.rule('[bold blue]Fine-tuning Clustered Model[/bold blue]')
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            param.requires_grad = False
    
    training_args = TrainingArguments(
        output_dir='./fine_tune_results',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        num_train_epochs=epochs,
        save_strategy='epoch',
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to='none',
        learning_rate=1e-5,
        weight_decay=0.01
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    
    for param in model.parameters():
        param.requires_grad = True
    
    ft_dir = './finetuned_clustered_model/'
    os.makedirs(ft_dir, exist_ok=True)
    model.save_pretrained(ft_dir)
    console.print(f'[bold green]Fine-tuned model saved to [cyan]{ft_dir}[/cyan][/bold green]')
    
    return model


def main():
    console.rule('[bold blue]Preparing Data[/bold blue]')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f'[bold]Using device:[/bold] [cyan]{device}[/cyan]')
    
    full_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_size = 5000
    val_size = 500
    train_data, remaining = random_split(full_train, [train_size, len(full_train) - train_size])
    val_data, _ = random_split(remaining, [val_size, len(remaining) - val_size])
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    train_dataset = CIFAR10Dataset(train_data, processor)
    eval_dataset = CIFAR10Dataset(val_data, processor)
    test_dataset = CIFAR10Dataset(test_data, processor)
    
    model_dir = './trained_model/'
    if os.path.exists(model_dir):
        console.rule('[bold blue]Loading Pretrained Model[/bold blue]')
        model = ViTForImageClassification.from_pretrained(model_dir).to(device)
        print_model_info(model, 'Pretrained ViT')
    else:
        console.rule('[bold blue]Training New Model[/bold blue]')
        model = train_model(train_dataset, eval_dataset, processor, device, epochs=5)
    
    console.rule('[bold blue]Clustering Weights[/bold blue]')
    clustered_model, centroids, indices, non_clustered = cluster_weights(model)
    
    clustered_dir = './clustered_model/'
    save_compressed_model(clustered_model, centroids, indices, clustered_dir)
    
    loaded_model = load_compressed_model(clustered_dir, device)
    
    console.print(f'[bold green]Compressed model saved to [cyan]{clustered_dir}[/cyan][/bold green]')
    console.print(f'[bold]Original size:[/bold] {sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024):.2f} MB')
    console.print(f'[bold]Compressed size:[/bold] {get_compressed_size(centroids, indices, non_clustered):.2f} MB')
    
    ft_clustered_model = fine_tune_clustered_model(clustered_model, train_dataset, eval_dataset, epochs=3)
    
    console.rule('[bold blue]Performance Evaluation[/bold blue]')
    
    original_metrics = measure_performance(model, test_dataset, device, 'Original ViT')

    clustered_metrics = measure_performance(
        loaded_model, test_dataset, device, 'Clustered ViT',
        centroids, indices, non_clustered
    )

    ft_metrics = measure_performance(
        ft_clustered_model, test_dataset, device, 'Fine-tuned Clustered ViT',
        centroids, indices, non_clustered
    )
    
    results = pd.DataFrame([original_metrics, clustered_metrics, ft_metrics])
    console.print('\n[bold]Performance Comparison:[/bold]')
    console.print(results)
    
    results.to_csv('clustering_results.csv', index=False)
    console.print('\n[bold green]Results saved to clustering_results.csv[/bold green]')

if __name__ == '__main__':
    main()