# This is a pytoch implementation of DISTS metric.
# Requirements: python >= 3.6, pytorch >= 1.0

import numpy as np
import os, sys
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import torch.fft

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class LearnableColorTransform(nn.Module):
    """Learnable color transformation network to improve color consistency."""
    def __init__(self):
        super(LearnableColorTransform, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, img):
        img_flat = img.view(-1, 3)  # Flatten to [num_pixels, 3]
        img_transformed = self.fc(img_flat)
        return img_transformed.view(img.shape)
    
class FourierCompression(nn.Module):
    """Learnable Fourier compression to reduce high-dimensional spectral features."""
    def __init__(self, in_channels, compressed_dim):
        super(FourierCompression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, compressed_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)  # Flatten

class STSIM_VGG(nn.Module):
    def __init__(self, dim, opt, grayscale=False):
        """
        Initialize STSIM_VGG with added features:
        - Learnable Color Transformation
        - Flip Padding for Edge Effects
        - CLIP Feature (Optional)
        - Fourier-based Loss (Optional)
        
        Args:
            dim (tuple): Dimensions for the linear layer.
            opt (dict): Options from configuration.
            grayscale (bool): Whether the input images are grayscale or RGB.
            use_clip (bool): Whether to integrate CLIP-based features.
            use_fourier (bool): Whether to use Fourier-based loss.
        """
        super(STSIM_VGG, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])        
        
        self.use_clip = opt['use_clip']
        self.use_fourier = opt['use_fourier']
        self.fourier_dim = opt['fourier_dim']
        self.use_color = opt['use_color']

        if self.use_clip:
            self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
            self.clip_model.to("cuda").eval()
            self.clip_feature_dim = 512  # CLIP feature size

        # Learnable color transformation
        if self.use_color:
            self.color_transform = LearnableColorTransform()

        # Fourier Compression Module
        if self.use_fourier:
            self.fourier_compression = FourierCompression(in_channels=1, compressed_dim=self.fourier_dim)

        # Mean and std for normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.C = 1e-10

        # Regularization options
        self.use_layer_norm = opt['train'].get('layer_norm', False)
        self.use_batch_norm = opt['train'].get('batch_norm', False)
        self.dropout_rate = opt['train'].get('dropout_rate', 0.0)

        # Linear mapping layer
        feature_dim = dim[0] + (self.clip_feature_dim if self.use_clip else 0) + (self.fourier_dim if self.use_fourier else 0)
        self.linear = nn.Linear(feature_dim, dim[1])

        # Add normalization layers
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(normalized_shape=feature_dim)
        elif self.use_batch_norm:
            self.norm = nn.BatchNorm1d(num_features=feature_dim)
        else:
            self.norm = None

        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else None

    def init_model(self):
        """Freeze VGG and CLIP feature extraction layers, train only custom layers."""
        for name, param in self.named_parameters():
            if "stage" in name or "clip" in name:  # Keep VGG & CLIP frozen
                param.requires_grad = False
            else:
                param.requires_grad = True  # Enable grad for trainable layers

        # Ensure Fourier and Color transform layers are trainable
        if self.use_fourier:
            for param in self.fourier_compression.parameters():
                param.requires_grad = True

        if self.use_color:
            for param in self.color_transform.parameters():
                param.requires_grad = True



    def flip_padding(self, x, ratio = 8):
        """Applies reflection padding to reduce edge artifacts."""
        _,_,H,W = x.shape
        return F.pad(x, (W//ratio, W//ratio, H//ratio, H//ratio), mode="reflect")
    def remove_padding(self, x, ratio = 8):
        _,_,H,W = x.shape
        pad_h = H // (ratio+2)
        pad_w = W // (ratio+2)
        return x[:,:,pad_h:-pad_h,pad_w:-pad_w]

    def extract_fourier_features(self, x):
        """Compute Log Magnitude Spectrum and compress."""
        x_freq = torch.fft.fft2(x, dim=(-2, -1))
        x_mag = torch.abs(x_freq)
        x_log_mag = torch.log(1 + x_mag)  # Log Magnitude Spectrum

        # Reduce to 4D by averaging frequency channels
        x_reduced = torch.mean(x_log_mag, dim=1, keepdim=True)  # Shape: [batch, 1, height, width]

        # Learnable Fourier Compression
        compressed_features = self.fourier_compression(x_reduced)  # Add channel dim
        return compressed_features


    def forward_once(self, x):
        """Extract hierarchical VGG-based texture features."""
        h = (x - self.mean) / self.std  # Normalize            
        if self.use_color:
            h = self.color_transform(h)  # Apply learnable color transformation
        h = self.flip_padding(h)  # Apply flip padding
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h

        coeffs = [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        f = []

        for c in coeffs:
            c = self.remove_padding(c)
            mean = torch.mean(c, dim=[2, 3])
            var = torch.var(c, dim=[2, 3])
            f.append(mean)
            f.append(var)

            c = c - mean.unsqueeze(-1).unsqueeze(-1)
            f.append(torch.mean(c[:, :, :-1, :] * c[:, :, 1:, :], dim=[2, 3]) / (var + self.C))
            f.append(torch.mean(c[:, :, :, :-1] * c[:, :, :, 1:], dim=[2, 3]) / (var + self.C))
        
        features = torch.cat(f, dim=-1)  # [BatchSize, FeatureSize]
        # print(f"feature_shape: {features.shape}")
        if self.use_fourier:
            fourier_features = self.extract_fourier_features(x)
            # print(f"Fourier feature_shape: {fourier_features.shape}")
            features = torch.cat([features, fourier_features.flatten(1)], dim=-1)
        if self.use_clip:
            clip_features = self.extract_clip_features(x)
            # print(f"clip feature_shape: {clip_feats0.shape}")
            features = torch.cat([features, clip_features], dim=-1)
        return features

    def extract_clip_features(self, x):
        """Extract CLIP features for additional guidance."""
        x_resized = F.interpolate(x, size=(224, 224), mode="bilinear")
        clip_features = self.clip_model.encode_image(x_resized)
        return clip_features

    def forward(self, x, y, require_grad=True):
        """Compute similarity between two inputs."""
        if require_grad:
            if len(x.shape) == 4:
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
            else:
                feats0, feats1 = x, y
        else:
            with torch.no_grad():
                if len(x.shape) == 4:
                    feats0 = self.forward_once(x)
                    feats1 = self.forward_once(y)
                else:
                    feats0, feats1 = x, y

        if self.use_layer_norm or self.use_batch_norm:
            feats0 = self.norm(feats0)
            feats1 = self.norm(feats1)

        if self.dropout is not None:
            feats0 = self.dropout(feats0)
            feats1 = self.dropout(feats1)

        pred = self.linear(torch.abs(feats0 - feats1))  # [N, dim]
        pred = torch.bmm(pred.unsqueeze(1), pred.unsqueeze(-1)).squeeze(-1)  # inner-prod

        return torch.sqrt(pred)  # [N, 1]

    @torch.no_grad()
    def inference(self, x, y):
        """Inference function with bias adjustment."""
        pred = self.forward(x, y)
        return pred - torch.sqrt(torch.sum(self.linear.bias**2))

def prepare_image(image, resize=True):
    if resize and min(image.size) > 256:
        image = transforms.functional.resize(image, 256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)


if __name__ == '__main__':
    from PIL import Image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='../images/r0.png')
    parser.add_argument('--dist', type=str, default='../images/r1.png')
    args = parser.parse_args()

    ref = prepare_image(Image.open(args.ref).convert("RGB"))
    dist = prepare_image(Image.open(args.dist).convert("RGB"))
    assert ref.shape == dist.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STSIM_VGG().to(device)
    import pdb;

    pdb.set_trace()
    ref = ref.to(device)
    dist = dist.to(device)
    score = model(ref, dist)
    print(score.item())
    # score: 0.3347