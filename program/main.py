import argparse
from pathlib import Path

IMAGE_SIZE = 512
DEFAULT_YOLO_WEIGHT_PATH = (
    Path(__file__).resolve().parent.parent / "Model" / "YOLOv9_Fold1.pt"
)
DEFAULT_MEDSAM2_WEIGHT_PATH = (
    Path(__file__).resolve().parent.parent / "Model" / "LoRA512_R32_Fold1.pt"
)


def run_inference(
    image_path: str | Path,
    output_mask_path: str | Path,
    fold_number: int = 1,
) -> Path:
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
