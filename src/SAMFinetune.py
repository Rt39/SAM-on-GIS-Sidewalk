import gdown
import os
import shutil
import sys
import numpy as np
import tifffile
from pycocotools.coco import COCO
from accelerate import Accelerator
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
from torch.optim import Adam
from monai.losses import DiceLoss
from tqdm import tqdm
import statistics

def download_data(data_path: str):
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
    return [f for f in os.listdir(img_dir) if (f.endswith('.tif') and np.max(tifffile.imread(os.path.join(label_dir, f))) > 0)]



def train_fn(model, epochs, learning_rate, plain_loader, prompt_loader, accelerator, checkpoint_path, model_using: str, resume_count: int = 0):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = DiceLoss(sigmoid=True, squared_pred=True)

    model, optimizer, plain_loader, prompt_loader = accelerator.prepare(model, optimizer, plain_loader, prompt_loader)

    for epoch in range(epochs):
        epoch_losses = []
        for batch in tqdm(plain_loader):
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

        print("Traing without prompt")
        print(f'Epoch {epoch+1}/{epochs}, Loss: {statistics.mean(epoch_losses)}')

        epoch_losses = []
        for batch in tqdm(prompt_loader):
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

        print("Traing with prompt")
        print(f'Epoch {epoch+1}, Loss: {statistics.mean(epoch_losses)}')

        # Save the model every epoch to avoid losing progress
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'finetune_sam_{model_using}_epoch_{(epoch+1+resume_count):03d}.pt'))

def evaluate_fn(model, val_dataloader_plain, val_dataloader_prompt):
    loss_fn = DiceLoss(sigmoid=True, squared_pred=True)

    val_losses = []
    for batch in tqdm(val_dataloader_plain):
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
    for batch in tqdm(val_dataloader_prompt):
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
    arg_parser.add_argument("-b", "--batch_size", type=int, default=2, help="Batch size for training.")
    arg_parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of epochs for training.")
    arg_parser.add_argument("-l", "--learning_rate", type=float, default=1e-5, help="Learning rate for training.")
    arg_parser.add_argument("-c", "--resume_training", action="store_true", help="Resume training from a checkpoint.")
    arg_parser.add_argument("--checkpoint_path", type=str, default=os.path.join("..", "models"), help="Path to save checkpoints.")
    arg_parser.add_argument("--data_path", type=str, default=os.path.join("..", "data"), help="Path to save data.")

    args = arg_parser.parse_args()


    # Get the model to use
    model_using = args.model
    if model_using not in ["base", "huge"]:
        print("Invalid model type. Please use either 'base' or 'huge'.")
        sys.exit()
    model_name = f"facebook/sam-vit-{model_using}"

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

    # Accelerator
    accelerator = Accelerator()

    # Load the data
    train_files = load_filter_data(train_path, train_label_path)
    val_files = load_filter_data(val_path, val_label_path)

    # Load model and processor
    sam_processor = SamProcessor.from_pretrained(model_name, cache_dir=args.data_path)
    sam_model = SamModel.from_pretrained(model_name, cache_dir=args.data_path)
    if args.resume_training:
        checkpoints = [os.path.join(args.checkpoint_path, f) for f in os.listdir(args.checkpoint_path) if f.startswith(f'finetune_sam_{model_using}_epoch_')]
        if not checkpoints:
            print("No checkpoint found to resume training. Using original model.")
        else:
            checkpoints.sort()
            sam_model.load_state_dict(torch.load(checkpoints[-1]))
            resume_count = int(checkpoints[-1].split('_')[-1].split('.')[0])

    # Create datasets and dataloaders
    train_dataset_plain = SidewalkDatasetPlain(train_path, train_label_path, train_files, sam_processor)
    train_dataset_prompt = SidewalkDatasetBboxPrompt(train_path, DVRPC_train_path, sam_processor)
    val_dataset_plain = SidewalkDatasetPlain(val_path, val_label_path, val_files, sam_processor)
    val_dataset_prompt = SidewalkDatasetBboxPrompt(val_path, DVRPC_val_path, sam_processor)

    train_dataloader_plain = DataLoader(train_dataset_plain, batch_size=args.batch_size, shuffle=True)
    train_dataloader_prompt = DataLoader(train_dataset_prompt, batch_size=args.batch_size, shuffle=True)
    val_dataloader_plain = DataLoader(val_dataset_plain, batch_size=args.batch_size, shuffle=False)
    val_dataloader_prompt = DataLoader(val_dataset_prompt, batch_size=args.batch_size, shuffle=False)

    # Train the model
    sam_model.to(accelerator.device).train()
    train_fn(sam_model, args.epochs, args.learning_rate, train_dataloader_plain, train_dataloader_prompt, accelerator, args.checkpoint_path, model_using, resume_count=resume_count if args.resume_training else 0)

    # Evaluate the model
    sam_model.eval()
    evaluate_fn(sam_model, val_dataloader_plain, val_dataloader_prompt)



if __name__ == '__main__':
    main()