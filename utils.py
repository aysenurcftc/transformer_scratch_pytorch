from pathlib import Path
import torch
from config import latest_weights_file_path, get_weights_file_path


def setup_device():
    """Sets up and returns the appropriate device."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS")
    else:
        device = "cpu"
        print("Using CPU")
    return torch.device(device)


def load_checkpoint(config, model, optimizer):
    """Loads checkpoint if specified in config."""
    initial_epoch = 0
    global_step = 0

    model_filename = (
        latest_weights_file_path(config)
        if config["preload"] == "latest"
        else get_weights_file_path(config, config["preload"])
    )

    if model_filename and Path(model_filename).exists():
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    return initial_epoch, global_step


def save_checkpoint(config, model, optimizer, epoch, global_step):
    """Saves model checkpoint."""
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        model_filename,
    )
