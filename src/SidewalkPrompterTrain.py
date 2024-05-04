import gdown
import os
import shutil
import sys
import numpy as np
import tifffile
from pycocotools.coco import COCO
from accelerate import Accelerator, DistributedDataParallelKwargs
import argparse
from SidewalkPrompter import *

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm
import statistics

# Accelerator
# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
accelerator = Accelerator()

# Download the data only on the main process to avoid data corruption
@accelerator.on_main_process
def download_data(data_path: str):
    """
    Downloads and extracts the necessary data for the project.

    Args:
        data_path (str): The path where the data will be downloaded and extracted.

    Returns:
        None
    """
    train_url = 'https://drive.google.com/file/d/1De6cOV0UtS310-vkILWpmY7hiJZRSU9Y/view?usp=drive_link'
    label_url = 'https://drive.google.com/file/d/1T8RDNBtxuBidm9ttNW9ShauDB49dBjWH/view?usp=drive_link'
    val_url = 'https://drive.google.com/file/d/1MFLm_5c0G6CUGNx2o2wrwGAKZHvUBCTI/view?usp=drive_link'
    DVRPC_train_url = 'https://drive.google.com/file/d/1pHzGmjQUvrH1TY4XL1vw8xg72u8K5BuI/view?usp=drive_link'
    DVRPC_val_url = 'https://drive.google.com/file/d/1YC5oUmGDa0sO14Qc4d-PM8cn2dbU1BKK/view?usp=drive_link'

    train_zip_path = os.path.join(data_path,'train.tar.gz')
    label_zip_path = os.path.join(data_path, 'label.tar.gz')
    val_zip_path = os.path.join(data_path,'val.tar.gz')
    DVRPC_train_path = os.path.join(data_path,'DVRPC_train.json')
    DVRPC_val_path = os.path.join(data_path,'DVRPC_val.json')

    train_path = os.path.join(data_path, 'Train')
    label_path = os.path.join(data_path, 'Label')
    val_path = os.path.join(data_path, 'Test')

    # Download and extract the data
    if not os.path.exists(train_path):
        gdown.download(train_url, train_zip_path, fuzzy=True, quiet=False, resume=True)
        shutil.unpack_archive(train_zip_path, data_path)
        os.remove(train_zip_path)   # Remove the zip file since it's too large
    if not os.path.exists(label_path):
        gdown.download(label_url, label_zip_path, fuzzy=True, quiet=False, resume=True)
        shutil.unpack_archive(label_zip_path, data_path)
        os.remove(label_zip_path)
        shutil.rmtree(os.path.join(label_path, 'Test2'))    # We don't need this folder
    if not os.path.exists(val_path):
        gdown.download(val_url, val_zip_path, fuzzy=True, quiet=False, resume=True)
        shutil.unpack_archive(val_zip_path, data_path)
        os.remove(val_zip_path)
    if not os.path.exists(DVRPC_train_path):
        gdown.download(DVRPC_train_url, DVRPC_train_path, fuzzy=True, quiet=False, resume=True)
    if not os.path.exists(DVRPC_val_path):
        gdown.download(DVRPC_val_url, DVRPC_val_path, fuzzy=True, quiet=False, resume=True)

def load_filter_data(img_dir: str, label_dir: str):
    global DEBUG
    file_names = []
    for i, f in enumerate(os.listdir(img_dir)):
        if DEBUG and i >= 10:
            break
        if f.endswith('.tif'):
            ground_truth = tifffile.imread(os.path.join(label_dir, f))
            if np.max(ground_truth) > 0:
                file_names.append(f)
    return file_names

# First loading model and processor may download, so we should
# do it on the main process first to avoid multiple downloads
# and data corruption
def load_model(checkpoint_path: str, resume_training: bool):
    model = SidewalkPrompter()
    resume_count = 0
    if resume_training:
        checkpoints = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if f.startswith(f'sidewalk_prompter_epoch_')]
        if not checkpoints:
            print("No checkpoint found to resume training. Using original model.")
        else:
            checkpoints.sort()
            model.load_state_dict(torch.load(checkpoints[-1]))
            resume_count = int(checkpoints[-1].split('_')[-1].split('.')[0])
    return model, resume_count

def train_fn(model, epochs: int, learning_rate, dataloader, checkpoint_path: str, resume_count: int = 0):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = LossFn()

    global checkpoint_name

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    # if resume_count > 0:
    #     print(f"Resuming training from epoch {resume_count}")
    #     accelerator.load_state(os.path.join(checkpoint_path, checkpoint_name.format(resume_count)))

    for epoch in range(epochs):
        epoch_losses = []
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            img, ground_truth = batch['image'], batch['ground_truth']
            optimizer.zero_grad()
            pred = model(img)
            l = loss_fn(pred, ground_truth)
            accelerator.backward(l)
            optimizer.step()
            epoch_losses.append(l.item())

        if accelerator.is_main_process:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {statistics.mean(epoch_losses)}')

            if (epoch+1) % 10 == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(checkpoint_path, checkpoint_name.format(epoch+1+resume_count)))

def evaluate_fn(model, dataloader):
    loss_fn = LossFn()

    model, dataloader = accelerator.prepare(model, dataloader)

    val_losses = []
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            img, ground_truth = batch['image'], batch['ground_truth']
            pred = model(img)
            l = loss_fn(pred, ground_truth)
            val_losses.append(l.item())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'Validation Loss: {statistics.mean(val_losses)}')

class SidewalkDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, files: list, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = tifffile.imread(os.path.join(self.img_dir, self.files[idx]))
        file_name = self.files[idx]
        img = np.moveaxis(img, -1, 0)
        label = tifffile.imread(os.path.join(self.label_dir, self.files[idx]))
        ground_truth = calculate_centroids(label)
        return {'image': torch.tensor(img).float(), 'ground_truth': torch.tensor(ground_truth).float(), 'file_name': file_name}


def main():
    arg_parser = argparse.ArgumentParser(
        prog="SidewalkPrompterTrain.py",
        description="Train the Sidewalk Prompter model.",
    )
    arg_parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training. (default: 32)")
    arg_parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs for training. (default: 100)")
    arg_parser.add_argument("-l", "--learning_rate", type=float, default=1e-5, help="Learning rate for training. (default: 1e-5)")
    arg_parser.add_argument("-c", "--resume_training", action="store_true", help="Resume training from a checkpoint.")
    arg_parser.add_argument("--checkpoint_path", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"), help="Path to save checkpoints.")
    arg_parser.add_argument("--data_path", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"), help="Path to save data.")
    arg_parser.add_argument("--debug", action="store_true")

    args = arg_parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    # Checkpoint name
    global checkpoint_name
    checkpoint_name = 'sidewalk_prompter_epoch_{}.pt'.format('{:04d}')

    # Prepare data
    os.makedirs(args.data_path, exist_ok=True)
    download_data(args.data_path)
    train_path = os.path.join(args.data_path, 'Train')
    label_path = os.path.join(args.data_path, 'Label')
    val_path = os.path.join(args.data_path, 'Test')
    train_label_path = os.path.join(label_path, 'Train')
    val_label_path = os.path.join(label_path, 'Test')

    # Load the data
    train_files = load_filter_data(train_path, train_label_path)
    val_files = load_filter_data(val_path, val_label_path)

    # Load model and processor
    model, resume_count = load_model(args.checkpoint_path, args.resume_training)

    # Create datasets and dataloaders
    train_dataset = SidewalkDataset(train_path, train_label_path, train_files)
    val_dataset = SidewalkDataset(val_path, val_label_path, val_files)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Train the model
    model.to(accelerator.device).train()
    train_fn(model, args.epochs, args.learning_rate, train_dataloader, args.checkpoint_path, resume_count=resume_count if args.resume_training else 0)

    # Evaluate the model
    model.eval()
    evaluate_fn(model, val_dataloader)



if __name__ == '__main__':
    main()