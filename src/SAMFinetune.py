import gdown
import os
import shutil
import sys
import numpy as np
import tifffile
from pycocotools.coco import COCO
from accelerate import Accelerator, DistributedDataParallelKwargs
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
from torch.optim import Adam
from monai.losses import DiceLoss
from tqdm.auto import tqdm
import statistics

# Accelerator
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

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
    """
    Load and filter data from the given image and label directories,
    returning only the files that its mask is not empty (i.e. has at least one pixel with value > 0)

    Args:
        img_dir (str): The directory path containing the images of sidewalks.
        label_dir (str): The directory path containing the masks of data.

    Returns:
        list: A list of file names contains sidewalks (i.e. its mask has at least one pixel with value > 0)

    """
    global DEBUG
    if DEBUG:
        ans = []
        for f in os.listdir(img_dir):
            if len(ans) > 10:
                return ans
            if not f.endswith('.tif'):
                continue
            mask = tifffile.imread(os.path.join(label_dir, f))
            if np.max(mask) > 0:
                ans.append(f)
    return [f for f in os.listdir(img_dir) if (f.endswith('.tif') and np.max(tifffile.imread(os.path.join(label_dir, f))) > 0)]

# First loading model and processor may download, so we should
# do it on the main process first to avoid multiple downloads
# and data corruption
def load_model_and_processor(model_using: str, data_path: str, checkpoint_path: str, resume_training: bool):
    """
    Load the SAM model and processor for fine-tuning.

    Args:
        model_using (str): The name of the model to use.
        data_path (str): The path to the data directory.
        checkpoint_path (str): The path to the checkpoint directory.
        resume_training (bool): Whether to resume training from a checkpoint.

    Returns:
        sam_model (SamModel): The loaded SAM model.
        sam_processor (SamProcessor): The loaded SAM processor.
        resume_count (int): The number of epochs to resume training from, only meaningful if resume_training is True.
    """
    model_name = f"facebook/sam-vit-{model_using}"

    with accelerator.main_process_first():
        sam_processor = SamProcessor.from_pretrained(model_name, cache_dir=data_path)
        sam_model = SamModel.from_pretrained(model_name, cache_dir=data_path)
        resume_count = 0
        if resume_training:
            checkpoints = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if f.startswith(f'finetune_sam_{model_using}_epoch_')]
            if not checkpoints:
                print("No checkpoint found to resume training. Using original model.")
            else:
                checkpoints.sort()
                # Load using accelerator
                # sam_model.load_state_dict(torch.load(checkpoints[-1]))
                resume_count = int(checkpoints[-1].split('_')[-1].split('.')[0])
        return sam_model, sam_processor, resume_count

def train_fn(model, epochs: int, learning_rate, plain_loader, prompt_loader, checkpoint_path: str, resume_count: int = 0):
    """
    Trains the model using the given data loaders for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to be trained.
        epochs (int): The number of epochs to train the model for.
        learning_rate (float): The learning rate for the optimizer.
        plain_loader (torch.utils.data.DataLoader): The data loader for plain images.
        prompt_loader (torch.utils.data.DataLoader): The data loader for images with bbox prompt.
        checkpoint_path (str): The path to save the model checkpoints.
        resume_count (int, optional): The count of resumed training. Defaults to 0.
    """
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = DiceLoss(sigmoid=True, squared_pred=True)

    global checkpoint_name

    model, optimizer, plain_loader, prompt_loader = accelerator.prepare(model, optimizer, plain_loader, prompt_loader)
    if resume_count > 0:
        print(f"Resuming training from epoch {resume_count}")
        accelerator.load_state(os.path.join(checkpoint_path, checkpoint_name.format(resume_count)))

    for epoch in range(epochs):
        epoch_losses = []
        for batch in tqdm(plain_loader, disable=not accelerator.is_local_main_process):
            # Forward pass
            outputs = model(pixel_values=batch['pixel_values'],
                                # input_boxes=batch['input_boxes'],
                                multimask_output=False)
            # Compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch['labels'].float()
            loss = loss_fn(predicted_masks, ground_truth_masks)

            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            epoch_losses.append(loss.item())

        print("Training without prompt")
        print(f'Epoch {epoch+1}/{epochs}, Loss: {statistics.mean(epoch_losses)}')

        epoch_losses = []
        for batch in tqdm(prompt_loader, disable=not accelerator.is_local_main_process):
            # Forward pass
            outputs = model(pixel_values=batch['pixel_values'],
                                input_boxes=batch['input_boxes'],
                                multimask_output=False)
            # Compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch['labels'].float()
            loss = loss_fn(predicted_masks, ground_truth_masks)

            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            epoch_losses.append(loss.item())

        print("Training with prompt")
        print(f'Epoch {epoch+1}, Loss: {statistics.mean(epoch_losses)}')

        # Save the model every epoch to avoid losing progress
        accelerator.wait_for_everyone()
        with accelerator.main_process_first():
            if accelerator.is_main_process:
                # Remove the previous checkpoint if it exists
                checkpoint_path = os.path.join(checkpoint_path, checkpoint_name.format(epoch+1+resume_count))
                if os.path.exists(checkpoint_path):
                    shutil.rmtree(checkpoint_path)
        accelerator.save_state(output_dir=os.path.join(checkpoint_path, checkpoint_name.format(epoch+1+resume_count)))
        # torch.save(model.state_dict(), os.path.join(checkpoint_path, checkpoint_name.format(epoch+1+resume_count)))

def evaluate_fn(model, val_dataloader_plain, val_dataloader_prompt):
    """
    Evaluate the model on the validation datasets.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        val_dataloader_plain (torch.utils.data.DataLoader): The dataloader for plain images.
        val_dataloader_prompt (torch.utils.data.DataLoader): The dataloader for images with bbox prompt.
    """
    loss_fn = DiceLoss(sigmoid=True, squared_pred=True)

    model, val_dataloader_plain, val_dataloader_prompt = accelerator.prepare(model, val_dataloader_plain, val_dataloader_prompt)

    val_losses = []
    for batch in tqdm(val_dataloader_plain, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(pixel_values=batch['pixel_values'],
                            multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch['labels'].float()
            loss = loss_fn(predicted_masks, ground_truth_masks.unsqueeze(1))
            val_losses.append(loss.item())

    print("Validation without prompt")
    print(f'Validation Loss: {statistics.mean(val_losses)}')

    val_losses = []
    for batch in tqdm(val_dataloader_prompt, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(pixel_values=batch['pixel_values'],
                            input_boxes=batch['input_boxes'],
                            multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch['labels'].float()
            loss = loss_fn(predicted_masks, ground_truth_masks.unsqueeze(1))
            val_losses.append(loss.item())

    print("Validation with prompt")
    print(f'Validation Loss: {statistics.mean(val_losses)}')


class SidewalkDatasetPlain(Dataset):
    def __init__(self, data_path: str, label_path: str, files: list, processor: SamProcessor, transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.files = files
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = tifffile.imread(os.path.join(self.data_path, self.files[idx]))
        label = tifffile.imread(os.path.join(self.label_path, self.files[idx]))
        if self.transform:
            img, label = self.transform(img, label)
        inputs = self.processor(img, return_tensors='pt')
        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs['labels'] = torch.tensor(label).unsqueeze(0)
        return inputs

class SidewalkDatasetBboxPrompt(Dataset):
    def __init__(self, data_path: str, coco_file: str, processor: SamProcessor, transform=None):
        self.data_path = data_path
        self.coco = COCO(coco_file)
        self.ann_ids = self.coco.getAnnIds()
        self.processor = processor
        self.transform = transform

    def __len__(self):
        global DEBUG
        if DEBUG:
            return 10
        return len(self.ann_ids)
    
    def __getitem__(self, idx):
        ann = self.coco.loadAnns(self.ann_ids[idx])[0]
        img = tifffile.imread(os.path.join(self.data_path, self.coco.imgs[ann['image_id']]['file_name']))
        label = self.coco.annToMask(ann)
        bbox = ann['bbox']
        if self.transform:
            img, label = self.transform(img, label)
        inputs = self.processor(img, input_boxes=[[bbox]], return_tensors='pt')
        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs['labels'] = torch.tensor(label).unsqueeze(0)
        return inputs


def main():
    arg_parser = argparse.ArgumentParser(
        prog="SAMFinetune",
        description="Finetune a pretrained SAM (Segment Anything Model) model on sidewalk data.",
    )
    arg_parser.add_argument("-m", "--model", type=str, default="base", help='Model to use for training, can be either "base" (using model "facebook/sam-vit-base") or "huge" (using model "facebook/sam-vit-base"). (default: base)')
    arg_parser.add_argument("-b", "--batch_size", type=int, default=2, help="Batch size for training. (default: 2)")
    arg_parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of epochs for training. (default: 5)")
    arg_parser.add_argument("-l", "--learning_rate", type=float, default=1e-5, help="Learning rate for training. (default: 1e-5)")
    arg_parser.add_argument("-c", "--resume_training", action="store_true", help="Resume training from a checkpoint.")
    arg_parser.add_argument("--checkpoint_path", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"), help="Path to save checkpoints.")
    arg_parser.add_argument("--data_path", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"), help="Path to save data.")
    arg_parser.add_argument("--debug", action="store_true")

    args = arg_parser.parse_args()

    global DEBUG
    DEBUG = args.debug


    # Get the model to use
    model_using = args.model
    if model_using not in ["base", "huge"]:
        print("Invalid model type. Please use either 'base' or 'huge'.")
        sys.exit()

    # Checkpoint name
    global checkpoint_name
    checkpoint_name = 'finetune_sam_{}_epoch_{}'.format(model_using, '{:03d}')

    # Prepare data
    os.makedirs(args.data_path, exist_ok=True)
    download_data(args.data_path)
    train_path = os.path.join(args.data_path, 'Train')
    label_path = os.path.join(args.data_path, 'Label')
    val_path = os.path.join(args.data_path, 'Test')
    DVRPC_train_path = os.path.join(args.data_path, 'DVRPC_train.json')
    DVRPC_val_path = os.path.join(args.data_path, 'DVRPC_val.json')
    train_label_path = os.path.join(label_path, 'Train')
    val_label_path = os.path.join(label_path, 'Test')

    # Load the data
    train_files = load_filter_data(train_path, train_label_path)
    val_files = load_filter_data(val_path, val_label_path)

    # Load model and processor
    sam_model, sam_processor, resume_count = load_model_and_processor(model_using, args.data_path, args.checkpoint_path, args.resume_training)

    # Create datasets and dataloaders
    train_dataset_plain = SidewalkDatasetPlain(train_path, train_label_path, train_files, sam_processor)
    train_dataset_prompt = SidewalkDatasetBboxPrompt(train_path, DVRPC_train_path, sam_processor)
    val_dataset_plain = SidewalkDatasetPlain(val_path, val_label_path, val_files, sam_processor)
    val_dataset_prompt = SidewalkDatasetBboxPrompt(val_path, DVRPC_val_path, sam_processor)

    train_dataloader_plain = DataLoader(train_dataset_plain, batch_size=args.batch_size, shuffle=True)
    train_dataloader_prompt = DataLoader(train_dataset_prompt, batch_size=args.batch_size, shuffle=True)
    val_dataloader_plain = DataLoader(val_dataset_plain, batch_size=args.batch_size, shuffle=False)
    val_dataloader_prompt = DataLoader(val_dataset_prompt, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Train the model
    sam_model.to(accelerator.device).train()
    train_fn(sam_model, args.epochs, args.learning_rate, train_dataloader_plain, train_dataloader_prompt, args.checkpoint_path, resume_count=resume_count if args.resume_training else 0)

    # Evaluate the model
    sam_model.eval()
    evaluate_fn(sam_model, val_dataloader_plain, val_dataloader_prompt)



if __name__ == '__main__':
    main()