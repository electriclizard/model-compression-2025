import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from rich.console import Console
from rich.table import Table
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from transformers import (Trainer, TrainingArguments,
                          ViTForImageClassification, ViTImageProcessor)

from dataset import CIFAR10Dataset
from utils import measure_performance

console = Console()


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_student = model(**inputs)
        loss_ce = outputs_student.loss
        logits_student = outputs_student.logits

        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits

        loss_kd = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(logits_student / self.args.temperature, dim=-1),
            F.softmax(logits_teacher / self.args.temperature, dim=-1)
        ) * (self.args.temperature ** 2)

        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd
        return (loss, outputs_student) if return_outputs else loss


def collate_fn(batch, feature_extractor):
    images = [x['img'] for x in batch]
    labels = torch.tensor([x['label'] for x in batch])
    inputs = feature_extractor(images, return_tensors='pt')
    return {'pixel_values': inputs['pixel_values'], 'labels': labels}


def print_model_info(model, name):
    table = Table(title=f"{name} Model Info")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    table.add_row("Total parameters", f"{total_params:,}")
    table.add_row("Trainable parameters", f"{trainable_params:,}")
    table.add_row("Device", next(model.parameters()).device.type)
    
    console.print(table)


def train_teacher_model(train_dataset, eval_dataset, processor, device):
    console.rule("[bold blue]Training Teacher Model[/bold blue]")
    
    with console.status("[bold green]Loading teacher model...[/bold green]"):
        teacher = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=10,
            ignore_mismatched_sizes=True
        ).to(device)
    
    print_model_info(teacher, "Teacher")
    
    teacher_args = TrainingArguments(
        output_dir='./teacher_results',
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
    
    teacher_trainer = Trainer(
        model=teacher,
        args=teacher_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
        
    console.print(f"[bold]Starting teacher training for {teacher_args.num_train_epochs} epochs[/bold]")
    teacher_trainer.train()
    
    teacher_dir = './trained_teacher/'
    os.makedirs(teacher_dir, exist_ok=True)
    teacher.save_pretrained(teacher_dir)
    processor.save_pretrained(teacher_dir)
    console.print(f"[bold green] Teacher model saved to [cyan]{teacher_dir}[/cyan][/bold green]")
    
    return teacher

def train_student_without_teacher(train_dataset, eval_dataset, processor, device):
    console.rule("[bold blue]Pretraining Student Model (Without Teacher)[/bold blue]")
    
    with console.status("[bold green]Loading student model...[/bold green]"):
        student = ViTForImageClassification.from_pretrained(
            'WinKawaks/vit-tiny-patch16-224',
            num_labels=10,
            ignore_mismatched_sizes=True
        ).to(device)
    
    print_model_info(student, "Student (Pretraining)")
    
    pretrain_args = TrainingArguments(
        output_dir='./student_pretrain_results',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        eval_strategy='epoch',
        save_strategy='epoch',
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        remove_unused_columns=False,
        report_to='none',
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01
    )
    
    pretrain_trainer = Trainer(
        model=student,
        args=pretrain_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
        
    console.print(f"[bold]Starting student pretraining for {pretrain_args.num_train_epochs} epochs[/bold]")
    pretrain_trainer.train()
    
    return student


def main():
    console.rule("[bold blue]Vision Transformer Distillation[/bold blue]")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"\n[bold]Using device:[/bold] [cyan]{device}[/cyan]")

    with console.status("[bold green]Loading processor...[/bold green]"):
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_rescale=False)

    console.rule("[bold blue]Loading Data[/bold blue]")
    with console.status("[bold green]Loading CIFAR-10 dataset...[/bold green]"):
        full_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = 5000
    val_size = 500

    console.print(f"\n[bold]Splitting dataset:[/bold]")
    console.print(f"- Training samples: [cyan]{train_size}[/cyan]")
    console.print(f"- Validation samples: [cyan]{val_size}[/cyan]")
    console.print(f"- Test samples: [cyan]{len(test_data)}[/cyan]")

    train_data, remaining = random_split(
        full_train,
        [train_size, len(full_train) - train_size]
    )
    val_data, _ = random_split(
        remaining,
        [val_size, len(remaining) - val_size]
    )

    with console.status("[bold green]Creating datasets[/bold green]"):
        train_dataset = CIFAR10Dataset(train_data, processor)
        eval_dataset = CIFAR10Dataset(val_data, processor)
        test_dataset = CIFAR10Dataset(test_data, processor)

    teacher = train_teacher_model(train_dataset, eval_dataset, processor, device)
    
    console.rule("[bold blue]Preparing Student Model[/bold blue]")
    
    student = train_student_without_teacher(train_dataset, eval_dataset, processor, device)
    
    pretrained_dir = './pretrained_student/'
    os.makedirs(pretrained_dir, exist_ok=True)
    student.save_pretrained(pretrained_dir)
    console.print(f"[bold green] Pretrained student model saved to [cyan]{pretrained_dir}[/cyan][/bold green]")
    
    result_folder = './distillation_results'
    os.makedirs(result_folder, exist_ok=True)

    training_args = DistillationTrainingArguments(
        output_dir=result_folder,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        eval_strategy='epoch',
        save_strategy='epoch',
        alpha=0.5,
        temperature=2.0,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        remove_unused_columns=False,
        report_to='none',
        learning_rate=5e-5,
    )

    console.rule("[bold blue]Starting Distillation Training[/bold blue]")
    console.print(f"\n[bold]Distillation parameters:[/bold]")
    console.print(f"- Alpha: [cyan]{training_args.alpha}[/cyan]")
    console.print(f"- Temperature: [cyan]{training_args.temperature}[/cyan]")
    console.print(f"- Epochs: [cyan]{training_args.num_train_epochs}[/cyan]")

    trainer = DistillationTrainer(
        model=student,
        args=training_args,
        teacher_model=teacher,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
 
    trainer.train()

    output_dir = './distilled_vit_tiny/'
    os.makedirs(output_dir, exist_ok=True)
    student.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    console.print(f"\n[bold green] Distilled student model saved to [cyan]{output_dir}[/cyan][/bold green]")

    console.rule("[bold blue]Evaluation[/bold blue]")
    with console.status("[bold green]Evaluating models[/bold green]"):
        pretrained_student = ViTForImageClassification.from_pretrained('./pretrained_student').to(device)
        
        teacher_metrics = measure_performance(teacher, test_dataset, device, model_type='Teacher')
        pretrained_student_metrics = measure_performance(
            pretrained_student,
            test_dataset,
            device,
            'Student (Pretrained only)'
        )
        distilled_student_metrics = measure_performance(
            student,
            test_dataset,
            device,
            'Student (After distillation)'
        )

    console.print("\n[bold]Evaluation Results:[/bold]")
    results = pd.DataFrame([teacher_metrics, pretrained_student_metrics, distilled_student_metrics])
    results.to_csv('./results_new.csv', index=False)


if __name__ == '__main__':
    main()