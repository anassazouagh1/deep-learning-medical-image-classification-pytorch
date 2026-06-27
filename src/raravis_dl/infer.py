import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD
from .models import build_model
from .utils import ensure_dir, get_device, load_checkpoint, load_config, save_json, set_seed


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


DEFAULT_CLASSES = [
    "Atelectasis",
    "Effusion",
    "Emphysema",
    "No finding",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with a trained PyTorch model for chest X-ray "
            "multi-class classification."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file used by the experiment.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to a single image for inference.",
    )

    parser.add_argument(
        "--image-dir",
        default=None,
        help="Path to a folder containing images for batch inference.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/inference",
        help="Directory where inference results will be saved.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to export.",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision during inference.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate inference input arguments."""
    has_image = args.image_path is not None
    has_folder = args.image_dir is not None

    if not has_image and not has_folder:
        raise ValueError("You must provide either --image-path or --image-dir.")

    if has_image and has_folder:
        raise ValueError("Use only one input option: --image-path or --image-dir.")


def safe_torch_load(path: str, map_location: str = "cpu") -> Any:
    """Load a PyTorch checkpoint in a version-compatible way."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_classes_from_checkpoint(checkpoint_path: str) -> List[str]:
    """Load class names from checkpoint metadata if available."""
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        classes = checkpoint.get("classes")

        if isinstance(classes, list) and len(classes) > 0:
            return [str(class_name) for class_name in classes]

        config = checkpoint.get("config")

        if isinstance(config, dict):
            data_config = config.get("data")

            if isinstance(data_config, dict):
                config_classes = data_config.get("classes")

                if isinstance(config_classes, list) and len(config_classes) > 0:
                    return [str(class_name) for class_name in config_classes]

    return DEFAULT_CLASSES


def build_inference_transform(image_size: int) -> transforms.Compose:
    """Build deterministic preprocessing for inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image_tensor(
    image_path: str,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Load an image and convert it into a normalized tensor."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError("Image not found: {}".format(path))

    image = Image.open(path).convert("RGB")
    transform = build_inference_transform(image_size)

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    return tensor


def discover_images(image_dir: str) -> List[Path]:
    """Discover all supported images inside a folder."""
    folder = Path(image_dir)

    if not folder.exists():
        raise FileNotFoundError("Image directory not found: {}".format(folder))

    images = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return sorted(images)


def build_model_from_config(
    config: Dict[str, Any],
    checkpoint_path: str,
    classes: List[str],
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Build a model from config and load checkpoint weights."""
    model = build_model(
        model_name=str(config["model_name"]),
        num_classes=len(classes),
        pretrained=False,
        dropout=float(config.get("dropout", 0.2)),
    )

    checkpoint_metadata = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint_metadata


def get_top_k_predictions(
    probabilities: torch.Tensor,
    classes: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Return top-k predictions as a list of dictionaries."""
    top_k = min(top_k, len(classes))

    values, indices = torch.topk(probabilities, k=top_k)

    results = []

    for rank, item in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        probability, class_index = item

        results.append(
            {
                "rank": int(rank),
                "class_index": int(class_index),
                "class_name": classes[int(class_index)],
                "probability": float(probability),
            }
        )

    return results


def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    classes: List[str],
    image_size: int,
    device: torch.device,
    top_k: int,
    use_amp: bool,
) -> Dict[str, Any]:
    """Run inference on one image and return prediction information."""
    image_tensor = load_image_tensor(
        image_path=image_path,
        image_size=image_size,
        device=device,
    )

    amp_enabled = bool(use_amp and device.type == "cuda")

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item())

    prediction = {
        "image_path": str(image_path),
        "file_name": Path(image_path).name,
        "predicted_index": predicted_index,
        "predicted_class": classes[predicted_index],
        "confidence": confidence,
        "top_predictions": get_top_k_predictions(
            probabilities=probabilities,
            classes=classes,
            top_k=top_k,
        ),
    }

    for class_index, class_name in enumerate(classes):
        safe_name = (
            class_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

        prediction["prob_{}".format(safe_name)] = float(
            probabilities[class_index].item()
        )

    return prediction


def flatten_predictions(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert prediction dictionaries into a clean CSV-friendly dataframe."""
    rows = []

    for prediction in predictions:
        row = {}

        for key, value in prediction.items():
            if key != "top_predictions":
                row[key] = value

        for top_prediction in prediction.get("top_predictions", []):
            rank = top_prediction["rank"]
            row["top{}_class".format(rank)] = top_prediction["class_name"]
            row["top{}_probability".format(rank)] = top_prediction["probability"]

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = ensure_dir(args.output_dir)

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    classes = load_classes_from_checkpoint(args.checkpoint)
    image_size = int(config["img_size"])

    device = get_device()
    use_amp = bool(config.get("use_amp", True)) and not args.no_amp

    print("=" * 80)
    print("Chest X-Ray Classification Inference")
    print("=" * 80)
    print("Config: {}".format(args.config))
    print("Checkpoint: {}".format(args.checkpoint))
    print("Output directory: {}".format(output_dir))
    print("Device: {}".format(device))
    print("Image size: {}".format(image_size))
    print("Number of classes: {}".format(len(classes)))
    print("=" * 80)

    model, checkpoint_metadata = build_model_from_config(
        config=config,
        checkpoint_path=args.checkpoint,
        classes=classes,
        device=device,
    )

    if args.image_path is not None:
        image_paths = [Path(args.image_path)]
    else:
        image_paths = discover_images(args.image_dir)

    if len(image_paths) == 0:
        raise ValueError("No valid images were found for inference.")

    predictions = []

    for image_path in image_paths:
        prediction = predict_single_image(
            model=model,
            image_path=str(image_path),
            classes=classes,
            image_size=image_size,
            device=device,
            top_k=int(args.top_k),
            use_amp=use_amp,
        )

        predictions.append(prediction)

        print(
            "{} -> {} ({:.4f})".format(
                Path(image_path).name,
                prediction["predicted_class"],
                prediction["confidence"],
            )
        )

    predictions_df = flatten_predictions(predictions)

    csv_path = output_dir / "inference_predictions.csv"
    json_path = output_dir / "inference_predictions.json"
    summary_path = output_dir / "inference_summary.json"

    predictions_df.to_csv(csv_path, index=False)
    save_json(predictions, json_path)

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "num_images": len(predictions),
        "image_size": image_size,
        "device": str(device),
        "classes": classes,
        "checkpoint_metadata": checkpoint_metadata,
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }

    save_json(summary, summary_path)

    print("\nFiles exported")
    print("- {}".format(csv_path))
    print("- {}".format(json_path))
    print("- {}".format(summary_path))

    print("\nInference finished successfully.")


if __name__ == "__main__":
    main()
