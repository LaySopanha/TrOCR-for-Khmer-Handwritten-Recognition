#!/usr/bin/env python3
"""
Gradio web demo that lets you choose between the trained TrOCR checkpoint and Tesseract.

Defaults:
- TrOCR weights: outputs/checkpoint-1900
- Processor/Tokenizer: khmer_trocr_model (or pass --processor-dir / --tokenizer-dir)
- Tesseract: lang khm, psm 6 (requires system Tesseract with Khmer traineddata)

Run:
  python scripts/web_demo.py
"""

import argparse
from pathlib import Path
from typing import Optional

import gradio as gr
import pytesseract
import torch
from PIL import Image
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "checkpoint-1900"
DEFAULT_PROCESSOR_DIR = PROJECT_ROOT / "khmer_trocr_model"

TROCR_MODEL = None
TROCR_PROCESSOR = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio web demo for Khmer OCR (TrOCR or Tesseract).")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory with TrOCR checkpoint.")
    parser.add_argument("--processor-dir", type=Path, default=DEFAULT_PROCESSOR_DIR, help="Directory with processor.")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Optional tokenizer directory (defaults to processor-dir).",
    )
    parser.add_argument("--tesseract-lang", type=str, default="khm", help="Tesseract language code (e.g., khm, khm+eng).")
    parser.add_argument("--tesseract-psm", type=int, default=6, help="Tesseract page segmentation mode.")
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Path to tesseract binary if not on PATH (e.g., /usr/bin/tesseract).",
    )
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Host for Gradio server.")
    parser.add_argument("--server-port", type=int, default=7860, help="Port for Gradio server.")
    return parser.parse_args()


def ensure_trocr_loaded(model_dir: Path, processor_dir: Path, tokenizer_dir: Optional[Path]) -> None:
    global TROCR_MODEL, TROCR_PROCESSOR
    if TROCR_MODEL is not None and TROCR_PROCESSOR is not None:
        return

    tokenizer_path = tokenizer_dir or processor_dir
    TROCR_PROCESSOR = TrOCRProcessor.from_pretrained(processor_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        fix_mistral_regex=True,
    )
    TROCR_PROCESSOR.tokenizer = tokenizer

    TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained(model_dir)
    TROCR_MODEL.to(DEVICE)
    TROCR_MODEL.eval()


def make_predict_fn(args):
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    tess_config = f"--psm {args.tesseract_psm}"

    def predict(image: Image.Image, backend: str) -> str:
        if image is None:
            return "No image provided."

        if backend == "TrOCR model":
            try:
                ensure_trocr_loaded(args.model_dir, args.processor_dir, args.tokenizer_dir)
            except Exception as exc:  # noqa: BLE001
                return f"Failed to load TrOCR model: {exc}"

            image_rgb = image.convert("RGB")
            with torch.inference_mode():
                pixel_values = TROCR_PROCESSOR(image_rgb, return_tensors="pt").pixel_values.to(DEVICE)
                generated_ids = TROCR_MODEL.generate(pixel_values)
                text = TROCR_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text

        # Tesseract path
        try:
            image_rgb = image.convert("RGB")
            text = pytesseract.image_to_string(image_rgb, lang=args.tesseract_lang, config=tess_config)
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            return f"Error (Tesseract): {exc}"

    return predict


def main() -> None:
    args = parse_args()
    predict = make_predict_fn(args)

    title = "Khmer Handwritten OCR – TrOCR or Tesseract"
    description = (
        "Upload a line image of Khmer handwriting and choose the backend. "
        f"TrOCR: {args.model_dir.name} | Tesseract: lang {args.tesseract_lang}, psm {args.tesseract_psm}"
    )

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Handwritten Line Image"),
            gr.Radio(
                choices=["TrOCR model", "Tesseract"],
                value="TrOCR model",
                label="Backend",
            ),
        ],
        outputs=gr.Textbox(label="Predicted Text"),
        title=title,
        description=description,
        flagging_mode="never",
    )

    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
