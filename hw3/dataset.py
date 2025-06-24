import torch


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Обработка изображения для ViT
        inputs = self.processor(images=image, return_tensors='pt')
        return {
            'pixel_values': inputs['pixel_values'][0],
            'labels': label
        }