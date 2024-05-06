from transformers import SamModel, SamProcessor
import torch
import numpy as np
import os
import sys

# Hack to import SidewalkPrompter from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")))
from SidewalkPrompter import SidewalkPrompter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_model.load_state_dict(torch.load(os.path.join("..", "..", "models", "finetune_sam_base_epoch_003.pt")))

sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

sidewalkPrompter = SidewalkPrompter()
sidewalkPrompter.load_state_dict(torch.load(os.path.join("..", "..", "models", "sidewalk_prompter_epoch_0060.pt")))

def predict_mask(image: np.ndarray) -> np.ndarray:
    inputs = sam_processor(image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        output = sam_model(pixel_values=pixel_values, multimask_output=False)
    return output.pred_masks.cpu().numpy().squeeze()

def predict_centroids(image: np.ndarray) -> np.ndarray:
    input_img = np.moveaxis(image, -1, 0)
    input_img = torch.tensor(input_img).float().unsqueeze(0)
    with torch.no_grad():
        output = sidewalkPrompter(input_img)
    return output.squeeze(0).cpu().numpy()

def parse_centroids(centroids: np.ndarray) -> np.ndarray:
    ans = centroids[centroids[:, :, 2] > 0.5]
    return np.round(np.clip(ans, 0, 256)).astype(int)

def predict_mask_with_centroids(image: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    inputs = sam_processor(image, input_points=[[centroids]], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    print(pixel_values.shape)
    input_points = inputs["input_points"].to(device)
    print(input_points.shape)
    with torch.no_grad():
        output = sam_model(pixel_values=pixel_values, input_points=input_points, multimask_output=False)
    return output.pred_masks.cpu().numpy().squeeze()