import os
import random
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import functional as TF
from metrics.STSIM import STSIM_M
import json

import re

def adjust_negative_path(path, adjustment=1):
    """
    Adjust the index in the file path by a specified adjustment value.

    Args:
        path (str): Original file path.
        adjustment (int): Value to adjust the index (default is +1).

    Returns:
        str: Updated file path with the corrected index.
    """
    # Use a regex to find the index in the filename
    match = re.search(r'(\d+)_th_texture', path)
    if not match:
        raise ValueError(f"Could not find an index to adjust in the path: {path}")

    # Extract the current index and calculate the new one
    current_index = int(match.group(1))
    new_index = current_index + adjustment

    # Replace the old index with the new index in the path
    updated_path = path.replace(f'{current_index}_th_texture', f'{new_index}_th_texture')
    return updated_path

def generate_perlin_noise(shape, scale=10):
    """
    Generate Perlin noise.

    Args:
        shape (tuple): Shape of the noise (height, width).
        scale (int): Scale of the Perlin noise.
    Returns:
        np.ndarray: Generated Perlin noise.
    """
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3  # Fade function

    lin_x, lin_y = np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0])
    grid_x, grid_y = np.meshgrid(lin_x, lin_y)
    grid_x = (grid_x * scale).astype(int)
    grid_y = (grid_y * scale).astype(int)

    random_gradients = np.random.randn(scale + 1, scale + 1, 2)
    dot_products = (
        (grid_x - grid_x.astype(int))[:, :, None] * random_gradients[grid_x, grid_y, 0] +
        (grid_y - grid_y.astype(int))[:, :, None] * random_gradients[grid_x, grid_y, 1]
    )
    t = f(grid_x - grid_x.astype(int))
    u = f(grid_y - grid_y.astype(int))

    return (1 - t) * (1 - u) * dot_products[:, :, 0] + \
            t * (1 - u) * dot_products[:, :, 1] + \
            (1 - t) * u * dot_products[:, :, 2] + \
            t * u * dot_products[:, :, 3]

class DynamicNegativeSampler:
    def __init__(self, opt, precalculated_path, name):
        """
        Args:
            opt (dict): Configuration options.
            precalculated_path (str): Path to the precalculated triplet data JSON file.
            patch_size (int): Size of the crops to be returned.
        """
        self.nega_real_ratio = opt['nega_real_ratio']
        self.patch_size = opt['datasets'][name]['patch_size']
        self.data_folder = opt['data_folder']

        # Load precalculated data
        if not os.path.exists(precalculated_path):
            raise FileNotFoundError(f"Precalculated data file not found: {precalculated_path}")

        with open(precalculated_path, 'r') as f:
            self.precalculated_data = json.load(f)

        # Image names from the precalculated data
        self.image_names = list(self.precalculated_data.keys())

    def get_negative_example(self, image_name, crop_index):
        """
        Get a negative crop example.

        Args:
            image_name (str): Name of the image for the query crop.
            crop_index (int): Index of the crop in the precalculated data.
        Returns:
            negative_crop (PIL.Image): The cropped negative example.
        """
        if random.random() < self.nega_real_ratio:
            # Choose a crop from a different image
            other_image_names = [img for img in self.image_names if img != image_name]
            if not other_image_names:
                raise ValueError(f"No valid negative images available excluding {image_name}.")

            # Randomly select another image and crop
            selected_image_name = random.choice(other_image_names)
            crop_data = random.choice(self.precalculated_data[selected_image_name])

            # Open the image and extract the crop
            image_path = os.path.join(self.data_folder, selected_image_name)
            image = Image.open(image_path).convert("RGB")
            left, top = crop_data["crop_position"]
            negative_crop = image.crop((left, top, left + self.patch_size, top + self.patch_size))
        else:
            # Use precalculated negatives
            negative_candidates = self.precalculated_data[image_name][crop_index]["negatives"]
            probabilities = [entry["probability"] for entry in negative_candidates]
            paths = [entry["path"] for entry in negative_candidates]

            # Sample one negative image based on probabilities
            sampled_index = random.choices(range(len(paths)), weights=probabilities, k=1)[0]
            sampled_path = paths[sampled_index]
            corrected_path = adjust_negative_path(sampled_path)

            # Open the image and perform a random crop
            image = Image.open(corrected_path).convert("RGB")
            left, top = self._get_random_crop_coordinates(image)
            negative_crop = image.crop((left, top, left + self.patch_size, top + self.patch_size))

        return negative_crop

    def _get_random_crop_coordinates(self, image):
        """
        Generate random crop coordinates for the given image.

        Args:
            image (PIL.Image): The image to generate crop coordinates for.
        Returns:
            (left, top) (tuple): Coordinates of the top-left corner of the crop.
        """
        width, height = image.size
        if width < self.patch_size or height < self.patch_size:
            raise ValueError(f"Image size ({width}, {height}) is smaller than patch size ({self.patch_size}).")

        left = random.randint(0, width - self.patch_size)
        top = random.randint(0, height - self.patch_size)
        return left, top

class TextureSampler(Dataset):
    def __init__(self, opt, sampler, precalculated_path, name):
        """
        Texture dataset with precalculated data and augmentations.

        Args:
            opt (dict): Dataset options.
            sampler (DynamicNegativeSampler): A sampler instance using precalculated data.
            precalculated_path (str): Path to the precalculated triplet data JSON file.
        """
        self.opt = opt
        self.patch_size = opt['datasets'][name]['patch_size']
        self.use_flip = opt['datasets'][name]['use_flip']
        self.use_rot = opt['datasets'][name]['use_rot']
        self.use_noise = opt['datasets'][name]['noise']['use_noise']
        self.noise_type = opt['datasets'][name]['noise']['noise_type'] if self.use_noise else None
        self.data_folder = opt['data_folder']

        # Load precalculated data
        if not os.path.exists(precalculated_path):
            raise FileNotFoundError(f"Precalculated data file not found: {precalculated_path}")
        
        with open(precalculated_path, 'r') as f:
            self.precalculated_data = json.load(f)

        # Image names from the precalculated data
        self.image_names = list(self.precalculated_data.keys())
        self.sampler = sampler

        # Initialize epoch
        self.epoch = 0  # Default epoch

    def set_epoch(self, epoch):
        """
        Update the current epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.epoch = epoch

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get image name and path
        image_name = self.image_names[idx]
        image_path = os.path.join(self.data_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        # Dynamically set a seed based on idx and epoch
        global_seed = self.opt.get('manual_seed', 0)
        random.seed(global_seed + self.epoch * 1000 + idx)  # Ensure variation across epochs and images

        # Select two crop indices for query and positive
        crop_indices = random.sample(range(len(self.precalculated_data[image_name])), 2)
        query_index, positive_index = crop_indices

        # Query crop
        query_data = self.precalculated_data[image_name][query_index]
        query_left, query_top = query_data["crop_position"]
        query_crop = image.crop((query_left, query_top, query_left + self.patch_size, query_top + self.patch_size))

        # Positive crop
        positive_data = self.precalculated_data[image_name][positive_index]
        positive_left, positive_top = positive_data["crop_position"]
        positive_crop = image.crop((positive_left, positive_top, positive_left + self.patch_size, positive_top + self.patch_size))

        # Negative crop
        negative_crop = self.sampler.get_negative_example(image_name, query_index)

        # Apply augmentations
        query_crop = self.apply_augmentations(query_crop)
        positive_crop = self.apply_augmentations(positive_crop)
        negative_crop = self.apply_augmentations(negative_crop)

        return query_crop, positive_crop, negative_crop

    def apply_augmentations(self, crop):
        """
        Apply flip, rotation, and noise to the given crop.

        Args:
            crop (PIL.Image): The crop to augment.
        Returns:
            crop (torch.Tensor): The augmented crop.
        """
        # Apply random horizontal and vertical flips
        if self.use_flip:
            if random.random() > 0.5:
                crop = ImageOps.mirror(crop)
            if random.random() > 0.5:
                crop = ImageOps.flip(crop)

        # Apply random rotation
        if self.use_rot:
            angle = random.choice([0, 90, 180, 270])  # Discrete rotations
            crop = crop.rotate(angle)

        # Apply noise
        if self.use_noise:
            crop = self.add_noise(crop)

        # Convert to tensor for PyTorch compatibility
        crop = TF.to_tensor(crop)
        return crop

    def add_noise(self, crop):
        """
        Add noise to the given crop based on the specified noise type.

        Args:
            crop (PIL.Image): The crop to add noise to.
        Returns:
            crop (PIL.Image): The crop with noise added.
        """
        crop_array = np.array(crop, dtype=np.float32) / 255.0  # Convert to [0, 1] range

        if self.noise_type == "gaussian":
            noise = np.random.normal(0, 0.1, crop_array.shape)  # Mean 0, Std 0.1
        elif self.noise_type == "salt_and_pepper":
            noise = np.random.choice([0, 1], size=crop_array.shape, p=[0.99, 0.01])
        elif self.noise_type == "Perlin":
            noise = generate_perlin_noise(crop_array.shape[:2])
            noise = np.expand_dims(noise, axis=-1)  # Expand dimensions to match channels
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

        noisy_crop = np.clip(crop_array + noise, 0, 1) * 255  # Clip and scale back to [0, 255]
        return Image.fromarray(noisy_crop.astype(np.uint8))
