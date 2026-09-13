import os
import random

from PIL import Image, ImageDraw


def create_synthetic_dataset(out_dir="data/processed", num_images=20):
    splits = {"train": int(num_images*0.7), "val": int(num_images*0.15), "test": int(num_images*0.15)}
    
    for split, count in splits.items():
        img_dir = os.path.join(out_dir, split, "images")
        lbl_dir = os.path.join(out_dir, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        for i in range(count):
            img = Image.new('RGB', (640, 640), color = (random.randint(50,200), random.randint(50,200), random.randint(50,200)))
            d = ImageDraw.Draw(img)
            
            lbl_lines = []
            # Add 2-4 random bounding boxes per image
            for _ in range(random.randint(2, 4)):
                cls_id = random.randint(0, 4)
                w, h = random.randint(50, 200), random.randint(50, 200)
                x, y = random.randint(0, 640-w), random.randint(0, 640-h)
                
                # Draw a rectangle just so it's not empty
                d.rectangle([x, y, x+w, y+h], outline=(255,255,255), width=3)
                
                # YOLO format: cls cx cy nw nh
                cx = (x + w/2) / 640.0
                cy = (y + h/2) / 640.0
                nw = w / 640.0
                nh = h / 640.0
                lbl_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
            base_name = f"{split}_{i}"
            img.save(os.path.join(img_dir, f"{base_name}.jpg"))
            with open(os.path.join(lbl_dir, f"{base_name}.txt"), "w") as f:
                f.write("\n".join(lbl_lines))

if __name__ == '__main__':
    create_synthetic_dataset()
    print("Synthetic dataset created.")
