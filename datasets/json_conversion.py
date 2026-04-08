import json
import os
import shutil
from tqdm import tqdm


def convert_flir_coco_to_yolo(json_path, output_labels_dir, target_class="person"):
    """
    Converts FLIR-ADAS-v2 coco.json annotations to YOLO .txt format with 1:1 symmetry.
    Ensures every image gets a label file, even if it is empty (background).
    """
    # 1. CLEAN START: Clear old labels to prevent appending errors
    if os.path.exists(output_labels_dir):
        shutil.rmtree(output_labels_dir)
    os.makedirs(output_labels_dir)

    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Map Category IDs
    cat_id_to_name = {cat['id']: cat['name'].lower() for cat in data['categories']}
    target_cat_ids = [cid for cid, name in cat_id_to_name.items() if name == target_class]

    if not target_cat_ids:
        print(f"Category '{target_class}' not found.")
        return

    target_id = target_cat_ids[0]

    # 2. Map all images to a dictionary to ensure we cover EVERY file
    images = {img['id']: img for img in data['images']}
    img_to_anns = {img_id: [] for img_id in images.keys()}

    # 3. Group annotations by image ID
    for ann in data['annotations']:
        if ann['category_id'] == target_id:
            img_to_anns[ann['image_id']].append(ann)

    print(f"Processing {len(images)} images for {target_class} symmetry...")

    for img_id, anns in tqdm(img_to_anns.items()):
        img_info = images[img_id]
        width = img_info['width']
        height = img_info['height']

        # Get clean base name (e.g., 'FLIR_0001')
        img_filename = os.path.basename(img_info['file_name'])
        base_name = os.path.splitext(img_filename)[0]
        txt_path = os.path.join(output_labels_dir, f"{base_name}.txt")

        # 4. Open in 'w' mode to create/overwrite
        with open(txt_path, 'w') as f_out:
            for ann in anns:
                # COCO format: [x_min, y_min, width, height]
                x_min, y_min, bw, bh = ann['bbox']

                # YOLO normalization (0-1)
                x_center = (x_min + bw / 2.0) / width
                y_center = (y_min + bh / 2.0) / height
                w_norm = bw / width
                h_norm = bh / height

                # Write YOLO line (Class 0 for person)
                f_out.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"Success: Verified 1:1 symmetry for {len(images)} images in {output_labels_dir}")


if __name__ == "__main__":
    # Training Set
    train_json = 'FLIR_ADAS_v2/images_thermal_train/coco.json'
    train_out = 'FLIR/labels/train/'
    convert_flir_coco_to_yolo(train_json, train_out)

    # Validation Set
    val_json = 'FLIR_ADAS_v2/images_thermal_val/coco.json'
    val_out = 'FLIR/labels/val/'
    convert_flir_coco_to_yolo(val_json, val_out)

    # Test Set
    test_json = 'FLIR_ADAS_v2/video_thermal_test/coco.json'
    test_out = 'FLIR/labels/test/'
    convert_flir_coco_to_yolo(test_json, test_out)