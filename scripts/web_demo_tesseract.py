#!/usr/bin/env python3
"""
Gradio web demo using Tesseract for Khmer handwritten OCR.

Prereqs:
- System Tesseract installed with Khmer traineddata (lang code: khm), e.g.:
  sudo apt-get install tesseract-ocr tesseract-ocr-khm
- Python: pip install -r requirements.txt (includes gradio + pytesseract)

Run:
  python scripts/web_demo_tesseract.py --lang khm --psm 6
"""

import argparse
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image
import pytesseract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio web demo for Tesseract OCR (Khmer).")
    parser.add_argument("--lang", type=str, default="khm", help="Tesseract language code (e.g., khm, khm+eng).")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode.")
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Path to tesseract binary if not on PATH (e.g., /usr/bin/tesseract).",
    )
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Host for Gradio server.")
    parser.add_argument("--server-port", type=int, default=7861, help="Port for Gradio server.")
    return parser.parse_args()


def make_predict_fn(lang: str, psm: int, tesseract_cmd: Optional[str]):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    config = f"--psm {psm}"

    def predict(image: Image.Image) -> str:
        if image is None:
            return "No image provided."
        try:
            image = image.convert("RGB")
            text = pytesseract.image_to_string(image, lang=lang, config=config)
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}"

    return predict


def main() -> None:
    args = parse_args()
    predict = make_predict_fn(args.lang, args.psm, args.tesseract_cmd)

    title = "Khmer Handwritten OCR (Tesseract)"
    description = (
        "Upload a line image and Tesseract will decode it using the specified language and PSM. "
        f"Lang: {args.lang}, PSM: {args.psm}"
    )

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Handwritten Line Image"),
        outputs=gr.Textbox(label="Predicted Text"),
        title=title,
        description=description,
        flagging_mode="never",
    )

    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
