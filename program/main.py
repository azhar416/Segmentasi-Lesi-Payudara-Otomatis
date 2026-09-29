import argparse
from pathlib import Path

IMAGE_SIZE = 512
DEFAULT_YOLO_WEIGHT_PATH = (
    Path(__file__).resolve().parent.parent / "Model" / "YOLOv9_Fold1.pt"
)
DEFAULT_MEDSAM2_WEIGHT_PATH = (
    Path(__file__).resolve().parent.parent / "Model" / "LoRA512_R32_Fold1.pt"
)


def ensure_runtime_dependencies() -> None:
    import importlib
    import importlib.util
    import re
    import subprocess
    import sys
    from importlib.metadata import PackageNotFoundError, version

    sam2_package = Path(__file__).resolve().parent / "sam2" / "sam2" / "__init__.py"
    if not sam2_package.is_file():
        raise FileNotFoundError(
            "SAM2 submodule is missing. From the project root run: "
            "git submodule update --init --recursive"
        )

    missing_torch = [
        module_name
        for module_name in ("torch", "torchvision")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_torch:
        raise RuntimeError(
            "Missing platform-specific dependencies: "
            f"{', '.join(missing_torch)}. Install the PyTorch build matching your "
            "CUDA/runtime environment before running inference."
        )

    try:
        torchao_version = version("torchao")
    except PackageNotFoundError:
        torchao_version = None

    if torchao_version is not None:
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", torchao_version)
        if match and tuple(map(int, match.groups())) <= (0, 16, 0):
            print(
                f"Removing incompatible optional torchao {torchao_version}; "
                "the installed PEFT requires torchao > 0.16.0 when it is present."
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"]
            )
            for module_name in tuple(sys.modules):
                if module_name == "torchao" or module_name.startswith("torchao."):
                    del sys.modules[module_name]

    module_to_package = {
        "cv2": "opencv-python-headless",
        "hydra": "hydra-core",
        "iopath": "iopath",
        "peft": "peft",
        "PIL": "Pillow",
        "ultralytics": "ultralytics",
    }
    missing_packages = [
        package_name
        for module_name, package_name in module_to_package.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_packages:
        print(f"Installing missing runtime dependencies: {', '.join(missing_packages)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing_packages]
        )
        importlib.invalidate_caches()


def run_inference(
    image_path: str | Path,
    output_mask_path: str | Path,
    fold_number: int = 1,
) -> Path:
    ensure_runtime_dependencies()

    import cv2
    from PIL import Image

    from program.model.medsam2_lora import MedSAM2LoRA
    from program.model.yolo import load_yolo
    from program.utils.utils import get_point_prompt, load_image, resize_image

    if not 1 <= fold_number <= 5:
        raise ValueError("fold_number must be between 1 and 5.")

    image_rgb = load_image(image_path)
    model_yolo = load_yolo(DEFAULT_YOLO_WEIGHT_PATH)
    results = model_yolo.predict(
        source=Image.fromarray(image_rgb),
        conf=0.25,
        iou=0.45,
        device="cpu",
        save=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show=False,
        stream=False,
        verbose=True,
    )

    bbox = None
    point_prompt = None
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        result = results[0]
        best_box_index = result.boxes.conf.argmax()
        bbox = result.boxes.xyxy[best_box_index].cpu().numpy()
        print(f"Bounding box: {bbox}")
    else:
        print("Tidak ada deteksi, menggunakan point prompt")
        point_prompt = get_point_prompt(fold_number)

    original_height, original_width = image_rgb.shape[:2]
    image_segmentasi = resize_image(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    if bbox is not None:
        scale_x = IMAGE_SIZE / original_width
        scale_y = IMAGE_SIZE / original_height
        bbox = bbox * [scale_x, scale_y, scale_x, scale_y]

    model_segmentasi = MedSAM2LoRA(
        weights_path=DEFAULT_MEDSAM2_WEIGHT_PATH,
        img_size=IMAGE_SIZE,
    )
    mask, _, _ = model_segmentasi.predict(
        image=image_segmentasi,
        bbox=bbox,
        point_prompt=point_prompt,
    )
    mask = cv2.resize(
        mask.astype("uint8"),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    output_mask_path = Path(output_mask_path)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output_mask_path), mask * 255)
    if not saved:
        raise OSError(f"Could not save output mask: {output_mask_path}")
    return output_mask_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO and MedSAM2-LoRA inference.")
    parser.add_argument("image_path", type=Path, help="Path to the input image")
    parser.add_argument("output_mask_path", type=Path, help="Path for the output mask")
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, 6),
        default=1,
        help="Fold used to select the fallback point prompt (default: 1)",
    )
    args = parser.parse_args()

    output_path = run_inference(
        image_path=args.image_path,
        output_mask_path=args.output_mask_path,
        fold_number=args.fold,
    )
    print(f"Mask saved to: {output_path}")


if __name__ == "__main__":
    main()
