import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as tf

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, EMNIST

from PIL import Image, UnidentifiedImageError

from utils.util import DEVICE, USE_CUDA


def make_permuted_mnist_dataloaders(batch_size, num_tasks):
    train_loaders = []
    test_loaders = []
    for task in range(num_tasks):
        train_dataset = MNIST("./data", train=True, download=True, transform=tf.ToTensor())
        test_dataset = MNIST("./data", train=False, download=True, transform=tf.ToTensor())

        train_data = train_dataset.data.float().div(255.).view(-1, 784).to(DEVICE)
        train_targets = train_dataset.targets.to(DEVICE)
        test_data = test_dataset.data.float().div(255.).view(-1, 784).to(DEVICE)
        test_targets = test_dataset.targets.to(DEVICE)

        perm = torch.randperm(784, device=DEVICE)
        train_data = train_data[:, perm]
        test_data = test_data[:, perm]

        train_loaders.append(data.DataLoader(data.TensorDataset(train_data, train_targets),
                                             batch_size, shuffle=True))
        test_loaders.append(data.DataLoader(data.TensorDataset(test_data, test_targets),
                                            batch_size, shuffle=False))

    return train_loaders, test_loaders

def make_split_mnist_dataloaders(batch_size, num_tasks):
    train_loaders = []
    test_loaders = []

    train_dataset = MNIST("./data", train=True, download=True, transform=tf.ToTensor())
    test_dataset = MNIST("./data", train=False, download=True, transform=tf.ToTensor())

    train_data = train_dataset.data.float().div(255.).view(-1, 784).to(DEVICE)
    train_targets = train_dataset.targets.to(DEVICE)
    test_data = test_dataset.data.float().div(255.).view(-1, 784).to(DEVICE)
    test_targets = test_dataset.targets.to(DEVICE)

    # sets of labels for each task
    sets_0 = [0, 2, 4, 6, 8]
    sets_1 = [1, 3, 5, 7, 9]

    for task in range(num_tasks):
        train_0_id = torch.where(train_targets == sets_0[task])[0]
        train_1_id = torch.where(train_targets == sets_1[task])[0]
        train_task_data = torch.cat((train_data[train_0_id], train_data[train_1_id]))
        train_task_targets = torch.cat((torch.zeros(train_0_id.shape[0]), torch.ones(train_1_id.shape[0])))

        test_0_id = torch.where(test_targets == sets_0[task])[0]
        test_1_id = torch.where(test_targets == sets_1[task])[0]
        test_task_data = torch.cat((test_data[test_0_id], test_data[test_1_id]))
        test_task_targets = torch.cat((torch.zeros(test_0_id.shape[0]), torch.ones(test_1_id.shape[0])))

        train_task_targets = train_task_targets.long().to(DEVICE)
        test_task_targets = test_task_targets.long().to(DEVICE)

        train_loaders.append(data.DataLoader(data.TensorDataset(train_task_data, train_task_targets),
                                             batch_size, shuffle=True))
        test_loaders.append(data.DataLoader(data.TensorDataset(test_task_data, test_task_targets),
                                            batch_size, shuffle=False))

    return train_loaders, test_loaders

class NotMNISTDataset(Dataset):
    def __init__(self, root_dir, letters, transform=None):
        self.root_dir = root_dir
        self.letters = letters
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for letter in self.letters:
            letter_dir = os.path.join(self.root_dir, letter)
            for filename in os.listdir(letter_dir):
                if filename.endswith(".png"):
                    image_paths.append(os.path.join(letter_dir, filename))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        try:
            image = Image.open(image_path).convert("L")  # Convert to grayscale
        except UnidentifiedImageError:  # Handle the case when the image cannot be identified
            print(f"Skipping unidentified image: {image_path}")
            return self.__getitem__(random.randint(0, len(self) - 1))  # Retry with a random image
        target = self.letters.index(os.path.basename(os.path.dirname(image_path)))
        if self.transform:
            image = self.transform(image)
        return image, target

def make_nomnist_dataloaders(batch_size, num_tasks):
    data_dir = "./data/notMNIST_small"
    train_loaders = []
    test_loaders = []
    sets_0 = ['A', 'B', 'C', 'D', 'E']
    sets_1 = ['F', 'G', 'H', 'I', 'J']
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    for task in range(num_tasks):
        train_letters = [sets_0[task], sets_1[task]]
        test_letters = [sets_0[task], sets_1[task]]
        train_dataset = NotMNISTDataset(data_dir, train_letters, transform=transform)
        test_dataset = NotMNISTDataset(data_dir, test_letters, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    return train_loaders, test_loaders

def make_split_cifar_dataloaders(batch_size, num_tasks):
    train_loaders = []
    test_loaders = []
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the input tensor
    ])

    num_classes = 10
    classes_per_task = num_classes // num_tasks

    for task in range(num_tasks):
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        
        if num_tasks == num_classes:
            classes = [task]
        else:
            classes = list(range(task * classes_per_task, (task + 1) * classes_per_task))

        train_indices = torch.tensor([i for i, label in enumerate(train_dataset.targets) if label in classes])
        test_indices = torch.tensor([i for i, label in enumerate(test_dataset.targets) if label in classes])

        if len(train_indices) > 0 and len(test_indices) > 0:
            train_dataset = data.Subset(train_dataset, train_indices)
            test_dataset = data.Subset(test_dataset, test_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
        else:
            print(f"Skipping task {task+1} due to insufficient samples.")

    return train_loaders, test_loaders

def make_fashionmnist_dataloaders(batch_size, num_tasks):
    train_loaders = []
    test_loaders = []
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize the image to 28x28
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the input tensor
    ])
    
    train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    
    num_classes = 10 
    classes_per_task = num_classes // num_tasks
    
    for task in range(num_tasks):
        task_classes = list(range(task * classes_per_task, (task + 1) * classes_per_task))
        task_train_indices = [i for i, t in enumerate(train_dataset.targets) if t in task_classes]
        task_test_indices = [i for i, t in enumerate(test_dataset.targets) if t in task_classes]
        task_train_dataset = data.Subset(train_dataset, task_train_indices)
        task_test_dataset = data.Subset(test_dataset, task_test_indices)
        train_loader = data.DataLoader(task_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(task_test_dataset, batch_size=batch_size, shuffle=False)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders

def fetch_datasets(batch_size, num_tasks, data_name='permuted_mnist'):
    if data_name == 'permuted_mnist':
        return make_permuted_mnist_dataloaders(batch_size, num_tasks)
    elif data_name == 'split_mnist':
        return make_split_mnist_dataloaders(batch_size, num_tasks)
    elif data_name == 'no_mnist':
        return make_nomnist_dataloaders(batch_size, num_tasks)
    elif data_name == 'split_cifar':
        return make_split_cifar_dataloaders(batch_size, num_tasks)
    elif data_name == 'fashion_mnist':
        return make_fashionmnist_dataloaders(batch_size, num_tasks)
    else:
        raise ValueError("Invalid data_name provided. Expected data_name in ['permuted_mnist', 'split_mnist', 'no_mnist', 'split_cifar'].")
