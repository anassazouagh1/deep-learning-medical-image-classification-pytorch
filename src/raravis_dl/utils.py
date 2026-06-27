from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """Set random seeds to improve experiment reproducibility.

    Full determinism is not always guaranteed in Deep Learning, especially when
    using CUDA kernels, but fixing the seed helps reduce uncontrolled variation
    between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a PyTorch file with compatibility across PyTorch versions.

    Some newer PyTorch versions changed checkpoint loading behaviour. This
    helper keeps loading compatible with checkpoints that contain metadata such
    as class names, configs and training information.
    """
    path = Path(path)

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def to_serializable(value: Any) -> Any:
    """Convert common Python, NumPy and PyTorch objects to JSON-safe values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_serializable(item) for item in value]

    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]

    return value


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(to_serializable(data), file, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain a dictionary: {path}")

    return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a dictionary: {path}")

    return data


def _get_nested(config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Read a value from a nested dictionary."""
    current: Any = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize configuration keys used by the training scripts.

    The repository supports both flat and nested YAML configuration styles.

    Flat example:
        model_name: efficientnet_b7
        img_size: 768
        batch_size: 8

    Nested example:
        model:
          name: efficientnet_b7
        data:
          image_size: 768
        training:
          batch_size: 8

    This function keeps the original values and also exposes the main settings
    as flat keys, so the rest of the pipeline can use a simple interface.
    """
    normalized = dict(config)

    normalized.setdefault(
        "experiment_name",
        config.get("experiment_name", "xray_classification_experiment"),
    )

    normalized.setdefault(
        "seed",
        _get_nested(config, ["experiment", "seed"], config.get("seed", 42)),
    )

    normalized.setdefault(
        "model_name",
        _get_nested(
            config,
            ["model", "name"],
            config.get("model_name", "efficientnet_b7"),
        ),
    )

    normalized.setdefault(
        "img_size",
        _get_nested(
            config,
            ["data", "image_size"],
            config.get("img_size", config.get("image_size", 224)),
        ),
    )

    normalized.setdefault(
        "batch_size",
        _get_nested(config, ["training", "batch_size"], config.get("batch_size", 8)),
    )

    normalized.setdefault(
        "epochs",
        _get_nested(config, ["training", "epochs"], config.get("epochs", 30)),
    )

    normalized.setdefault(
        "learning_rate",
        _get_nested(
            config,
            ["training", "learning_rate"],
            config.get("learning_rate", 1e-4),
        ),
    )

    normalized.setdefault(
        "weight_decay",
        _get_nested(
            config,
            ["training", "weight_decay"],
            config.get("weight_decay", 1e-4),
        ),
    )

    normalized.setdefault(
        "optimizer",
        _get_nested(config, ["training", "optimizer"], config.get("optimizer", "adamw")),
    )

    normalized.setdefault(
        "scheduler",
        _get_nested(config, ["training", "scheduler"], config.get("scheduler", "cosine")),
    )

    normalized.setdefault(
        "dropout",
        _get_nested(config, ["model", "dropout"], config.get("dropout", 0.2)),
    )

    normalized.setdefault(
        "use_pretrained",
        _get_nested(
            config,
            ["model", "pretrained"],
            config.get("use_pretrained", True),
        ),
    )

    normalized.setdefault(
        "num_workers",
        _get_nested(config, ["data", "num_workers"], config.get("num_workers", 4)),
    )

    normalized.setdefault(
        "use_amp",
        _get_nested(
            config,
            ["training", "mixed_precision"],
            config.get("use_amp", True),
        ),
    )

    normalized.setdefault(
        "use_class_weights",
        _get_nested(
            config,
            ["training", "use_class_weights"],
            config.get("use_class_weights", True),
        ),
    )

    normalized.setdefault(
        "use_weighted_sampler",
        _get_nested(
            config,
            ["training", "use_weighted_sampler"],
            config.get("use_weighted_sampler", False),
        ),
    )

    normalized.setdefault(
        "early_stopping_patience",
        _get_nested(
            config,
            ["training", "early_stopping_patience"],
            config.get("early_stopping_patience", 8),
        ),
    )

    normalized.setdefault(
        "monitor_metric",
        _get_nested(
            config,
            ["training", "monitor_metric"],
            config.get("monitor_metric", "f1_macro"),
        ),
    )

    normalized.setdefault(
        "scheduler_factor",
        _get_nested(
            config,
            ["training", "scheduler_factor"],
            config.get("scheduler_factor", 0.5),
        ),
    )

    normalized.setdefault(
        "scheduler_patience",
        _get_nested(
            config,
            ["training", "scheduler_patience"],
            config.get("scheduler_patience", 3),
        ),
    )

    normalized.setdefault(
        "min_learning_rate",
        _get_nested(
            config,
            ["training", "min_learning_rate"],
            config.get("min_learning_rate", 1e-6),
        ),
    )

    return normalized


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and normalize an experiment configuration file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        config = load_yaml(path)
    elif path.suffix.lower() == ".json":
        config = load_json(path)
    else:
        raise ValueError(
            f"Unsupported configuration format: {path.suffix}. "
            "Use .yaml, .yml or .json."
        )

    return normalize_config(config)


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    classes: list[str] | None = None,
    config: dict[str, Any] | None = None,
    epoch: int | None = None,
    best_score: float | None = None,
    monitor_metric: str | None = None,
    class_to_idx: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint with useful experiment metadata.

    The checkpoint stores more than model weights. It also includes class names,
    configuration and training metadata so the model can be evaluated later
    without losing the experiment context.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "config": config,
        "epoch": epoch,
        "best_score": best_score,
        "monitor_metric": monitor_metric,
        "torch_version": torch.__version__,
    }

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def _clean_state_dict_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove 'module.' prefix from DataParallel checkpoints if present."""
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[len("module.") :]] = value
        else:
            cleaned_state_dict[key] = value

    return cleaned_state_dict


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint into a model and return checkpoint metadata.

    The function supports:
    - Full checkpoints containing 'model_state_dict'.
    - Raw PyTorch state dictionaries.
    - DataParallel checkpoints with 'module.' prefixes.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key != "model_state_dict"
        }
    elif isinstance(checkpoint, dict) and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
        metadata = {}
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected either a dictionary with "
            "'model_state_dict' or a raw PyTorch state_dict."
        )

    cleaned_state_dict = _clean_state_dict_prefix(state_dict)
    model.load_state_dict(cleaned_state_dict, strict=strict)

    return metadata


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    """Append one dictionary as a JSON line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_serializable(row), ensure_ascii=False) + "\n")


def save_text(text: str, path: str | Path) -> None:
    """Save plain text to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        file.write(text)


def get_run_name(
    experiment_name: str,
    model_name: str,
    image_size: int,
    seed: int,
) -> str:
    """Create a clean experiment run name."""
    safe_experiment = experiment_name.lower().replace(" ", "_")
    safe_model = model_name.lower().replace(" ", "_")

    return f"{safe_experiment}_{safe_model}_{image_size}px_seed{seed}"


def print_config(config: dict[str, Any]) -> None:
    """Print a configuration dictionary in a readable way."""
    print(json.dumps(to_serializable(config), indent=2, ensure_ascii=False))


def count_files_by_extension(root: str | Path) -> dict[str, int]:
    """Count files by extension inside a directory."""
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    counts: dict[str, int] = {}

    for path in root.rglob("*"):
        if path.is_file():
            extension = path.suffix.lower() or "<no_extension>"
            counts[extension] = counts.get(extension, 0) + 1

    return dict(sorted(counts.items()))
