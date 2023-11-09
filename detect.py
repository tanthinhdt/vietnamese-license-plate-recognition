import os
import glob
import argparse
from tqdm import tqdm
from classes import LicensePlateRecognition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        help="Which modality to detect (image, video or realtime).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="The confidence threshold to filter out weak detections.",
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="The source of the input image or video.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="The destination of the output image or video.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    model = LicensePlateRecognition()
    model.load_from_checkpoint(
        detector_checkpoint_path="checkpoints/license_plate_detector.pt",
        ocr_checkpoint_path="checkpoints/license_plate_ocr.pt",
        confidence=args.confidence,
    )
    if args.modality == "realtime":
        model.inference(args.modality)
    elif os.path.isdir(args.src):
        file_paths = glob.glob(args.src + "/*")
        for file_path in tqdm(file_paths, desc="Inference", total=len(file_paths)):
            output_path = os.path.join(args.out, os.path.basename(file_path))
            model.inference(args.modality, file_path, output_path)
    else:
        model.inference(args.modality, args.src, args.out)


if __name__ == "__main__":
    main(parse_args())
