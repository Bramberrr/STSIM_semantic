import os
import random
from PIL import Image, ImageOps
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import functional as TF
from metrics.STSIM import STSIM_M, Metric
from torch.nn.functional import softmax

class TextureSampler(Dataset):
    def __init__(self, opt, sampler, image_paths, seed=0):
        """
        Texture dataset with dynamic negative sampling and augmentations.

        Args:
            opt (dict): Dataset options including patch size, augmentations, etc.
            sampler (DynamicNegativeSampler): A dynamic negative sampler instance.
            image_paths (list): List of image paths (split into training or validation set).
        """
        self.opt = opt
        self.patch_size = opt['patch_size']
        self.use_flip = opt['use_flip']
        self.use_rot = opt['use_rot']
        self.use_noise = opt['noise']['use_noise']
        self.noise_type = opt['noise']['noise_type'] if self.use_noise else None
        self.seed = seed

        # Image paths for the dataset
        self.image_paths = image_paths

        # Sampler for dynamic negatives
        self.sampler = sampler

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and dynamically sample a negative example
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Get query and positive crops from the same image
        query_crop, positive_crop = self.get_random_crop_pair(image)

        # Dynamically sample a negative example
        negative_path = self.sampler.get_negative_example(query_crop, image_path)
        negative_image = Image.open(negative_path).convert("RGB")
        negative_crop = self.get_random_crop(negative_image)

        # Apply augmentations
        query_crop = self.apply_augmentations(query_crop)
        positive_crop = self.apply_augmentations(positive_crop)
        negative_crop = self.apply_augmentations(negative_crop)

        return query_crop, positive_crop, negative_crop

    def get_random_crop_pair(self, image):
        """
        Get two distinct random crops of specified patch size from the same image.

        Args:
            image (PIL.Image): The image to crop.
        Returns:
            query_crop (PIL.Image): The first crop.
            positive_crop (PIL.Image): The second crop, distinct from the first.
        """
        width, height = image.size
        if width < self.patch_size or height < self.patch_size:
            raise ValueError(f"Image size ({width}, {height}) is smaller than patch size ({self.patch_size}).")

        # Ensure distinct crops by re-sampling if overlaps
        for _ in range(10):  # Retry up to 10 times
            left1 = random.randint(0, width - self.patch_size)
            top1 = random.randint(0, height - self.patch_size)
            left2 = random.randint(0, width - self.patch_size)
            top2 = random.randint(0, height - self.patch_size)

            # Check if the two crops are sufficiently distinct
            if abs(left1 - left2) > self.patch_size // 8 or abs(top1 - top2) > self.patch_size // 8:
                query_crop = image.crop((left1, top1, left1 + self.patch_size, top1 + self.patch_size))
                positive_crop = image.crop((left2, top2, left2 + self.patch_size, top2 + self.patch_size))
                return query_crop, positive_crop

        # Fallback to return two overlapping crops
        query_crop = image.crop((0, 0, self.patch_size, self.patch_size))
        positive_crop = image.crop((width - self.patch_size, height - self.patch_size, width, height))
        return query_crop, positive_crop

    def get_random_crop(self, image):
        """
        Get a random crop of specified patch size from the image.

        Args:
            image (PIL.Image): The image to crop.
        Returns:
            crop (PIL.Image): The cropped image.
        """
        width, height = image.size
        if width < self.patch_size or height < self.patch_size:
            raise ValueError(f"Image size ({width}, {height}) is smaller than patch size ({self.patch_size}).")

        left = random.randint(0, width - self.patch_size)
        top = random.randint(0, height - self.patch_size)
        crop = image.crop((left, top, left + self.patch_size, top + self.patch_size))
        return crop

    def apply_augmentations(self, crop):
        """
        Apply flip, rotation, and noise to the given crop.

        Args:
            crop (PIL.Image): The crop to augment.
        Returns:
            crop (PIL.Image or torch.Tensor): The augmented crop.
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
            noise = self.generate_perlin_noise(crop_array.shape[:2])
            noise = np.expand_dims(noise, axis=-1)  # Expand dimensions to match channels
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")

        noisy_crop = np.clip(crop_array + noise, 0, 1) * 255  # Clip and scale back to [0, 255]
        return Image.fromarray(noisy_crop.astype(np.uint8))
    def generate_perlin_noise(self, shape, scale=10):
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

        grid_x, grid_y = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))
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
    def __init__(self, opt, image_paths, device):
        """
        Initialize the DynamicNegativeSampler.

        Args:
            opt (dict): Configuration options.
            model_path (str): Path to the pretrained model.
            device (torch.device): Device to run computations on.
        """
        self.image_paths = image_paths
        self.gene_folder = opt['gene_folder']
        self.nega_real_ratio = opt['nega_real_ratio']
        self.feats_path = opt['feats_path']
        self.device = device

        # Load precomputed features for generated images
        self.target_feats = torch.load(self.feats_path).to(self.device)  # Pre-calculated features
        print(f"Loaded features of {len(self.target_feats)} images")
        self.target_imgfiles = sorted(os.listdir(self.gene_folder))

        # Load STSIM_M model for feature extraction
        self.model = STSIM_M([85, 10], mode=0, filter="SCF", device=device)
        self.model.load_state_dict(torch.load(opt['model_path'], map_location=self.device))
        self.model.to(device).double()

        self.m = Metric('SCF', self.device)

    def extract_feature(self, image):
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

        # Ensure the image has 3 channels
        if dist_img.shape[0] != 3:
            if dist_img.shape[0] == 1:
                dist_img = dist_img.repeat(3, 1, 1)  # Repeat channel for grayscale input
            elif dist_img.shape[0] > 3:
                dist_img = dist_img[:3]  # Use only the first 3 channels

        feat_color = []

    
        _, H, W = dist_img_gray.shape
        data = torch.zeros(3 * 3, 1, 256, 256)
        for i in range(3):
            for j in range(3):
                # Extract grayscale crop
                data[i * 3 + j, 0] = dist_img_gray[0, 
                                                   (H - 256) * i // 2:(H - 256) * i // 2 + 256,
                                                   (W - 256) * j // 2:(W - 256) * j // 2 + 256]
                # Extract color crop and compute mean for each channel
                data_color = dist_img[:, 
                                      (H - 256) * i // 2:(H - 256) * i // 2 + 256,
                                      (W - 256) * j // 2:(W - 256) * j // 2 + 256]
                feat_color.append(data_color.mean([1, 2]))

        # Compute STSIM-M features
        data = data.double().to(self.device)
        res = self.m.STSIM(data)

        # Add color features
        feat_color = torch.stack(feat_color).to(self.device)
        res = torch.cat([res, feat_color], dim=1)

        # Aggregate the features across all crops
        res = torch.mean(res, dim=0).reshape(1, 1, 1, -1)

        return res

    def get_negative_example(self, query_crop, image_name, N=30):
        """
        Dynamically sample a negative example based on the query crop.

        Args:
            query_crop (PIL.Image): The query crop for which the negative is selected.
            image_name (str): Path of the query image to exclude it from negatives.
            N (int): Number of top candidates to sample from.

        Returns:
            negative_crop (PIL.Image): Cropped negative example.
        """
        if random.random() < self.nega_real_ratio:
            # Select negative from image_paths excluding the current image
            negative_images = [img for img in image_paths if img != image_name]
            negative_image_path = random.choice(negative_images)
            negative_image = Image.open(negative_image_path).convert("RGB")
            negative_crop = self.get_random_crop(negative_image)
            return negative_crop
        else:
            # Extract feature for query crop
            query_feat = self.extract_feature(query_crop).to(self.device)

            # Compute distances to precomputed features
            tmp1 = query_feat.repeat(len(self.target_feats), 1, 1, 1)
            feats = self.target_feats.squeeze(1).squeeze(1).squeeze(1)
            dists = self.model(tmp1, feats)

            # Sort distances and get the top N indices
            sorted_dists, sorted_indices = torch.sort(dists.view(-1))
            top_N_indices = sorted_indices[:N]
            top_N_names = [self.target_imgfiles[idx] for idx in top_N_indices]

            # Normalize distances to probabilities
            probabilities = softmax(-sorted_dists[:N], dim=0).detach().cpu().numpy()

            # Sample one negative image based on probabilities
            sampled_index = random.choices(range(N), weights=probabilities, k=1)[0]
            sampled_image_path = os.path.join(self.gene_folder, top_N_names[sampled_index])

            sampled_image = Image.open(sampled_image_path).convert("RGB")
            negative_crop = self.get_random_crop(sampled_image)
            return negative_crop

    def get_random_crop(self, image):
        """
        Get a random crop from the image.

        Args:
            image (PIL.Image): The image to crop.

        Returns:
            crop (PIL.Image): The cropped image.
        """
        width, height = image.size
        left = random.randint(0, width - 256)
        top = random.randint(0, height - 256)
        crop = image.crop((left, top, left + 256, top + 256))
        return crop
