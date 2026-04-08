import cv2
import os
import random


def visualize_yolo_verification(image_dir, label_dir, output_name="verification_check.jpg"):
    # 1. Get list of all images in the directory
    # Supported formats for FLIR-v2 are typically .jpg [cite: 102, 103]
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print(f"❌ Error: No images found in {image_dir}")
        return

    # 2. Search for a random image that actually contains a 'person' (Class 0)
    # This avoids picking one of your ~2,500 background images
    random.shuffle(images)
    selected_img = None
    selected_label_data = []

    print("Searching for an image with a 'person' label...")
    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:  # Found a non-empty label file
                    selected_img = img_name
                    selected_label_data = lines
                    break

    if not selected_img:
        print("❌ Error: No images with 'person' labels found. Check if your .txt files are empty.")
        return

    # 3. Load the thermal image
    img_path = os.path.join(image_dir, selected_img)
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Error: Could not read image {img_path}")
        return

    height, width, _ = image.shape

    # 4. Draw the YOLO boxes
    for line in selected_label_data:
        parts = line.strip().split()
        if not parts: continue

        class_id = parts[0]
        # YOLO format is: class x_center y_center width height (all 0-1 normalized)
        x_c, y_c, w_norm, h_norm = map(float, parts[1:])

        # Denormalize coordinates to pixel values for OpenCV
        x1 = int((x_c - w_norm / 2) * width)
        y1 = int((y_c - h_norm / 2) * height)
        x2 = int((x_c + w_norm / 2) * width)
        y2 = int((y_c + h_norm / 2) * height)

        # Draw green box for 'person' (Class 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Person (Class {class_id})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. Save the result
    cv2.imwrite(output_name, image)
    print(f"✅ Success! Verification image saved as: {output_name}")
    print(f"Verified image: {selected_img} ({len(selected_label_data)} person(s) detected)")


if __name__ == "__main__":
    # Update these paths to match your successful conversion output
    # Based on your previous success message:
    train_images = 'FLIR/images/train/'
    train_labels = 'FLIR/labels/train/'

    visualize_yolo_verification(train_images, train_labels)