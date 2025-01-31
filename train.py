import os
import random
import logging
import shutil
import torch
from torch.utils.data import DataLoader, RandomSampler
import yaml
from datetime import datetime

from metrics.STSIM_VGG import STSIM_VGG
from utils import Prefetcher, precalculate_triplets
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import torch.multiprocessing as mp


def parse_options(yaml_path):
    """Parse options from YAML configuration."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def setup_experiment(opt):
    """
    Set up experiment folder and logging.

    Args:
        opt (dict): Configuration options containing experiment name.
    Returns:
        log_file (str): Path to the log file.
        model_dir (str): Path to the directory for saving models.
    """
    exp_dir = os.path.join("experiments", opt['name'])
    model_dir = os.path.join(exp_dir, "models")
    log_file = os.path.join(exp_dir, "train.log")

    # If the experiment folder exists, rename it with its creation time
    if os.path.exists(exp_dir):
        created_time = datetime.fromtimestamp(os.path.getctime(exp_dir)).strftime("%Y-%m-%d_%H-%M-%S")
        backup_dir = f"{exp_dir}_{created_time}"
        shutil.move(exp_dir, backup_dir)
        print(f"Renamed existing experiment folder to: {backup_dir}")

    # Create necessary directories
    os.makedirs(model_dir, exist_ok=True)
    return log_file, model_dir, exp_dir


def init_logger(log_file):
    """
    Initialize a logger for training.

    Args:
        log_file (str): Path to the log file.
    """
    logger = logging.getLogger("STSIM_Training")
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file, mode="a")

    # Set formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def split_dataset(data_folder, split_ratio):
    """
    Split dataset into training and validation sets based on the split ratio.

    Args:
        data_folder (str): Path to the dataset folder containing images.
        split_ratio (list): Ratio for training and validation split (e.g., [0.8, 0.2]).
    Returns:
        train_paths (list): List of training image paths.
        val_paths (list): List of validation image paths.
    """
    all_images = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder)])
    random.shuffle(all_images)

    split_idx = int(len(all_images) * split_ratio[0])
    train_paths = all_images[:split_idx]
    val_paths = all_images[split_idx:]
    return train_paths, val_paths


def save_triplet(q, p, n, output_path, iteration):
    """Save the first query, positive, and negative crops as images."""
    output_dir = os.path.join(output_path, "triplets")
    os.makedirs(output_dir, exist_ok=True)

    to_pil = ToPILImage()
    q_img = to_pil(q)
    p_img = to_pil(p)
    n_img = to_pil(n)

    q_img.save(os.path.join(output_dir, f"{iteration}_query.png"))
    p_img.save(os.path.join(output_dir, f"{iteration}_positive.png"))
    n_img.save(os.path.join(output_dir, f"{iteration}_negative.png"))


def plot_losses(train_losses, val_losses, output_path, opt):
    """
    Plot training and validation losses.

    Args:
        train_losses (list): List of recorded training losses.
        val_losses (list): List of recorded validation losses.
        output_path (str): Path to save the plot.
        opt (dict): Configuration dictionary containing logging frequencies.
    """
    plt.figure(figsize=(10, 6))

    # Generate x-axis values for training and validation losses
    train_iters = [i * opt['logger']['print_freq'] for i in range(len(train_losses))]
    val_iters = [i * opt['val']['val_freq'] for i in range(len(val_losses))]

    # Plot losses
    plt.plot(train_iters, train_losses, label="Training Loss", marker='o', linestyle='-')
    plt.plot(val_iters, val_losses, label="Validation Loss", marker='x', linestyle='--')

    # Labels and title
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(output_path, "loss_plot.png"))
    print(f"Loss plot saved to {os.path.join(output_path, 'loss_plot.png')}")


def train(opt):
    # Setup experiment directories and logger
    log_file, model_dir, exp_dir = setup_experiment(opt)
    logger = init_logger(log_file)
    logger.info(opt)
    # Set random seeds
    random.seed(opt['manual_seed'])
    torch.manual_seed(opt['manual_seed'])

    # Split dataset into training and validation sets
    train_paths, val_paths = split_dataset(opt['data_folder'], opt['datasets']['split_ratio'])
    logger.info(f"Training images: {len(train_paths)} | Validation images: {len(val_paths)}")
    use_old = False
    # Initialize the dynamic negative sampler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_old:
        from data.datasets_old import TextureSampler, DynamicNegativeSampler
        train_sampler = DynamicNegativeSampler(opt=opt, image_paths=train_paths, device=device)
        val_sampler = DynamicNegativeSampler(opt=opt, image_paths=val_paths, device=device)

        # Initialize TextureSampler datasets
        train_dataset = TextureSampler(opt['datasets']['train'], train_sampler, train_paths)
        val_dataset = TextureSampler(opt['datasets']['val'], val_sampler, val_paths)
    else:
        from data.datasets import TextureSampler, DynamicNegativeSampler
        # Paths for saving precalculated triplets
        train_precalc_file = os.path.join(opt['precalculated_path'], f"precalculated_triplets_train_{opt['num_nega_candi']}_{opt['datasets']['split_ratio'][0]}.json")
        val_precalc_file = os.path.join(opt['precalculated_path'], f"precalculated_triplets_val_{opt['num_nega_candi']}_{opt['datasets']['split_ratio'][1]}.json")

        # Precalculate triplets for training and validation sets if files don't exist
        if not os.path.exists(train_precalc_file):
            precalculate_triplets(
                opt=opt,
                model_path=opt['model_path'],
                image_paths=train_paths,
                split_name='train',
                num_crops=opt['datasets']['train']['num_crops'],
                N=opt['num_nega_candi']
            )
            logger.info(f"Precalculated file saved to {train_precalc_file}")
        else:
            logger.info(f"Load precalculated file from {train_precalc_file}")
        if not os.path.exists(val_precalc_file):
            precalculate_triplets(
                opt=opt,
                model_path=opt['model_path'],
                image_paths=val_paths,
                split_name='val',
                num_crops=opt['datasets']['val']['num_crops'],
                N=opt['num_nega_candi']
            )
            logger.info(f"Precalculated file saved to {val_precalc_file}")
        else:
            logger.info(f"Load precalculated file from {val_precalc_file}")

        train_sampler = DynamicNegativeSampler(opt, train_precalc_file, name='train')
        val_sampler = DynamicNegativeSampler(opt, val_precalc_file, name='val')

        train_dataset = TextureSampler(opt, train_sampler, train_precalc_file, name='train')
        val_dataset = TextureSampler(opt, val_sampler, val_precalc_file, name='val')

    # Create Random Sampler
    train_random_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(opt['manual_seed']))
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        sampler=train_random_sampler,
        batch_size=opt['train']['batch_size'],
        # shuffle=opt['datasets']['train']['use_shuffle'],
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt['val']['batch_size'],
        shuffle=opt['datasets']['val']['use_shuffle'],
        num_workers=0,
        pin_memory=True
    )

    # Prefetchers
    train_prefetcher = Prefetcher(train_loader)
    val_prefetcher = Prefetcher(val_loader)

    # Initialize the STSIM_VGG model
    model = STSIM_VGG(dim=(5900, 10), opt=opt).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt['train']['optimizer']['lr'],
        weight_decay=opt['train']['optimizer']['weight_decay'],
        betas=opt['train']['optimizer']['betas']
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt['train']['scheduler']['T_max'],
        eta_min=opt['train']['scheduler']['eta_min']
    )
    train_losses = []
    val_losses = []

    margin = opt['train']['margin']

    model.init_model()
    
    # Training loop
    model.train()

    val_prefetcher.reset()
    epoch = 0
    global_step = 0
    try:
        while global_step < opt['train']['total_iter']:
            logger.info(f"Starting Epoch {epoch + 1}")
            train_prefetcher.reset()

            while True:
                batch = train_prefetcher.next()
                if batch is None or global_step == opt['train']['total_iter']:
                    break

                query, positive, negative = [x.to(device) for x in batch]

                optimizer.zero_grad()

                pos_dist = model(query, positive)
                neg_dist = model(query, negative)

                logger.info(f"Requires Grad? pos_dist: {pos_dist.requires_grad}, neg_dist: {neg_dist.requires_grad}")

                logger.info(f"Mean Positive Distance: {pos_dist.mean().item():.4f}, Std: {pos_dist.std().item():.4f}")
                logger.info(f"Mean Negative Distance: {neg_dist.mean().item():.4f}, Std: {neg_dist.std().item():.4f}")

                loss = torch.mean(torch.relu(pos_dist - neg_dist + margin))

                logger.info(f"Loss Requires Grad? {loss.requires_grad}")  # Should be True

                for name, param in model.named_parameters():
                    print(f"{name} requires_grad: {param.requires_grad}")  # Debugging check
                loss.backward()
                # print(loss.item())

                # Gradient clipping
                if opt['train']['use_grad_clip']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()
                scheduler.step()

                # Logging
                if global_step % opt['logger']['print_freq'] == 0:
                    logger.info(f"Step {global_step}/{opt['train']['total_iter']} | Loss: {loss.item():.4f}")
                    train_losses.append(loss.item())

                # Visualization
                if global_step % opt['train']['visualization_freq'] == 0:
                    save_triplet(query[0].cpu(), positive[0].cpu(), negative[0].cpu(), exp_dir, global_step)

                # Validation
                if (global_step+1) % opt['val']['val_freq'] == 0 or global_step == 0:
                    model.eval()
                    val_loss = 0
                    val_batch = val_prefetcher.next()
                    if val_batch is None:
                        val_prefetcher.reset()
                        val_batch = val_prefetcher.next()
                        continue
                    
                    with torch.no_grad():
                        query, positive, negative = [x.to(device) for x in val_batch]
                        pos_dist = model(query, positive)
                        neg_dist = model(query, negative)
                        val_loss = torch.mean(torch.relu(pos_dist - neg_dist + margin)).item()

                    logger.info(f"Validation Loss at Step {global_step}: {val_loss:.4f}")
                    val_losses.append(val_loss)
                    model.train()

                # Save checkpoint
                if (global_step+1) % opt['logger']['save_checkpoint_freq'] == 0:
                    checkpoint_path = os.path.join(model_dir, f"stsim_checkpoint_{global_step}.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info(f"Checkpoint saved at Step {global_step}: {checkpoint_path}")

                global_step += 1
            epoch += 1

    except Exception as e:
        logger.error(f"Exception during training: {e}")
        raise

    # Plot losses after training
    plot_losses(train_losses, val_losses, exp_dir, opt)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    opt = parse_options("options/stsim.yml")
    train(opt)
