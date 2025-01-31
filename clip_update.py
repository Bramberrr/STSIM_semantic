import os
import json
import torch
import open_clip
from torchvision import transforms
from PIL import Image
from torch.nn.functional import softmax
from tqdm import tqdm
import yaml
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


def load_precalculated_file(file_path):
    """Load the precalculated triplets JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_updated_file(data, output_path):
    """Save the updated triplets to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated triplets saved to {output_path}")

def compute_clip_similarity(query_crop, candidate_paths, clip_model, preprocess, device):
    """Compute CLIP similarities between the query crop and candidate images."""
    query_tensor = preprocess(query_crop).unsqueeze(0).to(device)

    similarities = []
    for candidate_path in candidate_paths:
        candidate_image = Image.open(candidate_path).convert("RGB")
        candidate_tensor = preprocess(candidate_image).unsqueeze(0).to(device)

        # Compute features and cosine similarity
        with torch.no_grad():
            query_features = clip_model.encode_image(query_tensor)
            candidate_features = clip_model.encode_image(candidate_tensor)

            query_features /= query_features.norm(dim=-1, keepdim=True)
            candidate_features /= candidate_features.norm(dim=-1, keepdim=True)

            similarity = (query_features @ candidate_features.T).item()
        similarities.append(similarity)

    return similarities



def update_triplets_with_clip(precalculated_file, output_file, opt_path):
    """
    Update the negatives and probabilities in the precalculated triplets JSON using CLIP similarities.

    Args:
        precalculated_file (str): Path to the precalculated triplets JSON file.
        output_file (str): Path to save the updated triplets JSON file.
        opt_path (str): Path to the options JSON file.
    """
    # Load options from YAML file
    with open(opt_path, 'r') as f:
        try:
            opt = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error loading YAML file {opt_path}: {e}")

    clip_top_ratio = opt.get("clip_top_ratio", 0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion400m_e32')
    clip_model.to(device)
    clip_model.eval()

    # Load the precalculated triplets
    with open(precalculated_file, 'r') as f:
        triplets = json.load(f)

    updated_triplets = {}

    # Update each triplet with CLIP similarities
    for image_name, crops in tqdm(triplets.items(), desc="Processing Triplets"):
        updated_crops = []

        for crop_data in crops:
            crop_position = crop_data["crop_position"]
            negatives = crop_data["negatives"]

            # Load the query crop image
            query_image = Image.open(os.path.join(opt["data_folder"], image_name)).convert("RGB")
            left, top = crop_position
            patch_size = opt["datasets"]["train"]["patch_size"]
            query_crop = query_image.crop((left, top, left + patch_size, top + patch_size))
            query_crop_tensor = preprocess(query_crop).unsqueeze(0).to(device)

            # Compute CLIP similarities
            similarities = []
            for negative in negatives:
                # Correct the negative path by adjusting the index
                original_path = negative["path"]
                try:
                    corrected_path = adjust_negative_path(original_path, adjustment=1)  # Adjust index by +1
                    negative_image = Image.open(corrected_path).convert("RGB")  # Load image from corrected path
                except Exception as e:
                    print(f"Error loading image: {corrected_path}, falling back to original path. Error: {e}")
                    corrected_path = original_path
                    negative_image = Image.open(corrected_path).convert("RGB")
                
                # Preprocess and compute similarity
                negative_crop_tensor = preprocess(negative_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    query_feat = clip_model.encode_image(query_crop_tensor).float()
                    negative_feat = clip_model.encode_image(negative_crop_tensor).float()
                    similarity = torch.cosine_similarity(query_feat, negative_feat).item()

                similarities.append(similarity)

                # Update the path in the negative dictionary
                negative["path"] = corrected_path
                updated_negatives.append(negative)

            # Filter top candidates based on clip_top_ratio
            num_candidates = max(1, int(len(similarities) * clip_top_ratio))
            top_indices = torch.topk(torch.tensor(similarities), num_candidates).indices.tolist()

            updated_negatives = [negatives[i] for i in top_indices]
            top_similarities = [similarities[i] for i in top_indices]

            # Normalize similarities to probabilities
            probabilities = torch.softmax(torch.tensor(top_similarities), dim=0).tolist()

            for i, prob in enumerate(probabilities):
                updated_negatives[i]["probability"] = prob

            updated_crops.append({
                "crop_position": crop_position,
                "negatives": updated_negatives
            })

        updated_triplets[image_name] = updated_crops

    # Save the updated triplets to a new JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(updated_triplets, f, indent=4)
    print(f"Updated triplets saved to {output_file}")

if __name__ == "__main__":
    precalculated_file = "/home/pappas/STSIM_semantic/precalculated/precalculated_triplets_train_30_0.8.json"
    output_file = "/home/pappas/STSIM_semantic/precalculated/precalculated_triplets_with_clip_train_30_0.8.json"

    opt_path = "/home/pappas/STSIM_semantic/options/stsim.yml"

    update_triplets_with_clip(precalculated_file, output_file, opt_path)

