import os
import json
import random
from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional as TF
from metrics.STSIM import STSIM_M, Metric
from torch.nn.functional import softmax
from torchvision import transforms
from tqdm import tqdm

class Prefetcher:
    """
    Data prefetcher to load data batches asynchronously, improving data loading efficiency.
    """
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.next_input = None
        self.preload()

    def preload(self):
        try:
            self.next_input = next(iter(self.loader))
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = [x.cuda(non_blocking=True) for x in self.next_input]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_input = self.next_input
        self.preload()
        return next_input

    def reset(self):
        self.next_input = None
        self.preload()


def extract_feature(image, m, device):
    """
    Extract features for the given image using STSIM-M with color mode and data split logic.
    Args:
        image (PIL.Image): The input image as a PIL.Image object.
    Returns:
        torch.Tensor: Extracted features.
    """
    # Resize the image to ensure consistent dimensions
    transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])
    image = transform(image)

    # Convert the image to grayscale and tensor
    gray_image = ImageOps.grayscale(image)  # Convert to grayscale
    dist_img_gray = TF.to_tensor(gray_image)

    # Convert the image to tensor (RGB)
    dist_img = TF.to_tensor(image)

    if dist_img.dim() is 3:
        dist_img = dist_img.unsqueeze(0)

    if dist_img_gray.dim() is 3:
        dist_img_gray = dist_img_gray.unsqueeze(0)

    # print(dist_img_gray.dim())
    feat_color = dist_img.mean([2,3]).to(device)

    # Compute STSIM-M features
    dist_img_gray = dist_img_gray.double().to(device)
    res = m.STSIM(dist_img_gray)

    res = torch.cat([res, feat_color], dim=1)

    return res

def precalculate_triplets(opt, model_path, image_paths, split_name, num_crops, N=10):
    """
    Precalculate crop positions and top N negatives with probabilities for each image.

    Args:
        opt (dict): Dataset options, including paths and configurations.
        model_path (str): Path to the STSIM model.
        output_path (str): Path to save the precalculated JSON file.
        image_paths (list): List of image paths (train/val set).
        split_name (str): Name of the split (e.g., 'train' or 'val').
        num_crops (int): Number of crops to generate per image.
    """
    gene_folder = opt['gene_folder']
    feats_path = opt['feats_path']
    patch_size = opt['datasets'][split_name]['patch_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load precomputed features for generated images
    target_feats = torch.load(feats_path).to(device)
    target_imgfiles = sorted(os.listdir(gene_folder))

    # Load STSIM_M model
    model = STSIM_M([85, 10], mode=0, filter="SCF", device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).double()

    m = Metric('SCF', device)

    precalculated_data = {}

    # Process each image
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Generate random crops
        crop_positions = []
        for _ in range(num_crops):
            left = random.randint(0, width - patch_size)
            top = random.randint(0, height - patch_size)
            crop_positions.append((left, top))

        # For each crop, calculate top N negatives
        crop_data = []
        for left, top in crop_positions:
            crop = image.crop((left, top, left + patch_size, top + patch_size))
            # crop_tensor = TF.to_tensor(crop).to(device).unsqueeze(0).double()

            feat = extract_feature(crop, m, device)
            tmp1 = feat.repeat([len(target_feats), 1, 1, 1]).squeeze(1).squeeze(1).to(dtype=torch.float64)
            feats = target_feats.squeeze(1).squeeze(1).squeeze(1).to(dtype=torch.float64)
            dists = model(tmp1, feats)

            # Sort distances and get the top N indices
            sorted_dists, sorted_indices = torch.sort(dists.view(-1))
            top_N_indices = sorted_indices[:N]
            top_N_dists = sorted_dists[:N]
            top_N_names = [target_imgfiles[idx+1] for idx in top_N_indices]

            # Normalize distances to probabilities
            probabilities = softmax(-top_N_dists, dim=0).detach().cpu().numpy()

            crop_data.append({
                "crop_position": (left, top),
                "negatives": [
                    {"path": os.path.join(gene_folder, top_N_names[i]), "probability": probabilities[i]}
                    for i in range(N)
                ]
            })

        precalculated_data[image_name] = crop_data

    # Save to JSON
    os.makedirs(opt['precalculated_path'], exist_ok=True)
    output_file = os.path.join(opt['precalculated_path'], f"precalculated_triplets_{split_name}_{opt['num_nega_candi']}_{opt['datasets']['split_ratio'][0] if split_name == 'train' else opt['datasets']['split_ratio'][1]}.json")
    with open(output_file, "w") as f:
        json.dump(precalculated_data, f, indent=4)
    print(f"Precalculated triplets for {split_name} saved to {output_file}")
