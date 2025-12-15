import csv 
import json
import os
import unicodedata

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# configurations
JSON_PATH = "../data/annotation/annotated_data.json"
SOURCE_IMG_DIR = "../data/image/"
OUTPUT_DIR = "../data/dataset"
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
CSV_PATH = os.path.join(OUTPUT_DIR, "labels.csv")


def normalize_text(text: str) -> str:
    """Basic Khmer label normalization."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    for ch in ["\u200b", "\u200c", "\u200d", "\ufeff"]:
        text = text.replace(ch, "")
    return text.strip()

def preprocess_data():
    os.makedirs(CROPS_DIR, exist_ok=True)

    print(f"Loading annotations from {JSON_PATH}...")
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: JSON file not found.")
        return
    
    csv_data = []
    total_crops = 0

    print("Preprocessing images, masking background, and cropping...")

    for entry in tqdm(data):
        image_path_raw = entry['data']['image']
        image_name = os.path.basename(image_path_raw)
        full_image_path = os.path.join(SOURCE_IMG_DIR, image_name)

        if not os.path.exists(full_image_path):
            tqdm.write(f"Warning: Image {full_image_path} not found, skipping...")
            continue
        try:
            original_image = Image.open(full_image_path).convert("RGB")
            img_w, img_h = original_image.size
        except Exception as e:
            tqdm.write(f"Error opening image {image_name}: {e}")
            continue

        # map id to data
        id_to_text = {}
        id_to_point = {}

        for annotation in entry['annotations']:
            for result in annotation['result']:
                result_id = result['id']
                result_type = result['type']

                if result_type == 'textarea':
                    if 'text' in result['value'] and len(result['value']['text']) > 0:
                        cleaned = normalize_text(result['value']['text'][0])
                        if cleaned:
                            id_to_text[result_id] = cleaned
                elif result_type == 'polygonlabels' or result_type == 'rectanglelabels':
                    id_to_point[result_id] = result['value']['points']
        
        # crop and save
        for item_id, text in id_to_text.items():
            if item_id in id_to_point:
                points = id_to_point[item_id]

                # convert percentage points to absolute pixels (List of Tuples for PIL)
                pixel_points_tuples = []
                # Also keep a numpy array for min/max calculation
                pixel_points_np = []
                
                for p in points:
                    px = (p[0] / 100) * img_w
                    py = (p[1] / 100) * img_h
                    pixel_points_tuples.append((px, py))
                    pixel_points_np.append([px, py])
                
                pixel_points_np = np.array(pixel_points_np)
                
                if len(pixel_points_tuples) < 2: 
                    continue

                # Calculate the Bounding Box
                min_x = max(0, int(np.min(pixel_points_np[:, 0])))
                max_x = min(img_w, int(np.max(pixel_points_np[:, 0])))
                min_y = max(0, int(np.min(pixel_points_np[:, 1])))
                max_y = min(img_h, int(np.max(pixel_points_np[:, 1])))

                # REMOVED PADDING to reduce overlap risk
                pad = 0 
                crop_box = (
                    max(0, min_x - pad),
                    max(0, min_y - pad),
                    min(img_w, max_x + pad),
                    min(img_h, max_y + pad)
                )

                # Create a Mask for the Polygon
                # Create a black image same size as original
                mask = Image.new('L', (img_w, img_h), 0)
                draw = ImageDraw.Draw(mask)
                # Draw the polygon in white
                draw.polygon(pixel_points_tuples, outline=255, fill=255)

                #Crop the Image and the Mask
                try:
                    cropped_image_raw = original_image.crop(crop_box)
                    cropped_mask = mask.crop(crop_box)

                    if cropped_image_raw.width < 5 or cropped_image_raw.height < 5:
                        continue

                    # Apply the Mask (Make background white)
                    # Create a white background
                    final_image = Image.new('RGB', cropped_image_raw.size, (255, 255, 255))
                    # Paste the raw crop onto the white background, using the polygon mask
                    final_image.paste(cropped_image_raw, (0, 0), mask=cropped_mask)

                    # Save crop
                    crop_filename = f"{os.path.splitext(image_name)[0]}_{item_id}.jpg"
                    crop_save_path = os.path.join(CROPS_DIR, crop_filename)
                    final_image.save(crop_save_path)

                    csv_data.append([crop_filename, text])
                    total_crops += 1
                except Exception as e:
                    tqdm.write(f"Error cropping {item_id}: {e}")

    # write csv 
    print(f"Saving CSV to {CSV_PATH}...")
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        writer.writerows(csv_data)
    print(f"Success! Processed {len(data)} pages. Created {total_crops} text line images.")

if __name__ == "__main__":
    preprocess_data()
