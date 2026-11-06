import random
import time
import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display

# --- 1. Path and Model Configuration ---
# Input image folder
# Ensure this path is relative to the location of your Notebook file
IMAGES_DIR = Path("Malaysia-road-sign-dataset-1/test/images")
# Output folder
OUT_DIR = Path("predictions_right_panel")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Model Path
# Ensure this path is correct.
MODEL_PATH = "runs/detect/train2/weights/best.pt"

# --- 2. Decision Rule Dictionary---
DECISION_RULES = {
    'Bicycle lane': 'Yield to cyclists',
    'Bumps': 'Reduce speed for bumps',
    'Bumps ahead': 'Slow down; bumps ahead',
    'Bus stop': 'Do not block; watch pedestrians',
    'Camera operation zone': 'Obey speed limit; cameras active',
    'Chevron -left-': 'Sharp left curve; steer carefully',
    'Chevron -right-': 'Sharp right curve; steer carefully',
    'Children': 'Slow down; children crossing',
    'Construction sign': 'Reduce speed; construction zone',
    'Cow nearby': 'Watch for animals on road',
    'Crossroad': 'Prepare to stop or yield',
    'Crossroad on the left': 'Intersection from left; be ready',
    'Crossroad on the right': 'Intersection from right; be ready',
    'Crosswind area': 'Hold steering; crosswinds possible',
    'Double Bend to Left Ahead': 'Two left bends; slow down',
    'Double Bend to Right Ahead': 'Two right bends; slow down',
    'Expressway signs 1': 'Entering expressway; maintain speed',
    'Expressway signs 2': 'Stay in lane; high-speed traffic',
    'Flagman ahead': 'Prepare to stop; flagman ahead',
    'Give way': 'Yield to other vehicles',
    'Height limit': 'Do not enter if over height limit',
    'Horn Prohibited': 'Do not honk',
    'Left Bend Ahead': 'Left curve; reduce speed',
    'Level crossing with gates ahead': 'Prepare to stop; railway crossing',
    'Motorcycles only': 'Motorcycles only; do not enter',
    'Narrow bridge': 'Bridge narrows; proceed with caution',
    'No Stopping': 'No stopping at any time',
    'No U-turns': 'U-turn prohibited',
    'No entry': 'Do not enter; wrong way',
    'No left turn': 'Left turn prohibited',
    'No overtaking': 'Do not overtake',
    'No parking': 'Parking prohibited',
    'No right turn': 'Right turn prohibited',
    'Obstruction': 'Road blocked; change lane safely',
    'Other dangers nearby': 'Drive cautiously; hazard ahead',
    'Parking area': 'Parking allowed',
    'Pass either side': 'Pass left or right of obstacle',
    'Pass on the left': 'Pass only on the left',
    'Pass on the right': 'Pass only on the right',
    'Right Bend Ahead': 'Right curve; reduce speed',
    'Road cones': 'Follow cone guidance; lane shift',
    'Road narrows on the left': 'Narrows left; stay right',
    'Road narrows on the right': 'Narrows right; stay left',
    'Road work': 'Reduce speed; workers ahead',
    'Roadway diverges': 'Road splits; choose correct lane',
    'Roundabout ahead': 'Yield at roundabout',
    'Slippery road': 'Avoid sudden moves; slippery road',
    'Speed limit': 'Set speed to posted limit',
    'Stop': 'Apply brakes; full stop',
    'T-junction': 'T-intersection; yield or stop',
    'Towing area': 'No illegal parking; towing enforced',
    'Traffic from Left Merges Ahead': 'Allow space for left merge',
    'Traffic from Right Merges Ahead': 'Allow space for right merge',
    'Traffic merging from the left': 'Vehicles entering left; be alert',
    'Traffic merging from the right': 'Vehicles entering right; be alert',
    'Traffic merging to the left': 'Lane merges left; prepare to change',
    'Traffic signals ahead': 'Be ready to stop at lights',
    'Train Gate': 'Stop if gate closed',
    'U turn': 'U-turn permitted',
    'Weight limit': 'Do not enter if over weight limit',
    'Zebra crossing': 'Stop for pedestrians',
    'pedestrian crossing opt1': 'Slow down; pedestrian crossing'
}

def main():
    # --- 3. Loading Models and Images ---
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not IMAGES_DIR.exists():
        print(f"Error: Directory {IMAGES_DIR} not found.")
        return

    imgs = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in extensions])

    if not imgs:
        print(f"No images found in {IMAGES_DIR}")
        return

    # Randomly sample 10 images (if fewer than 10 are available, take all)
    num_samples = min(10, len(imgs))
    sampled = random.sample(imgs, num_samples)
    print(f"Selected {num_samples} images for processing...")

    # Attempt to load the font once (in preparation for calculating text width)
    font_path = None
    try:
        font_path = "arial.ttf" # Assuming Arial is available
        ImageFont.truetype(font_path, size=30)
    except IOError:
        print("Warning: Arial font not found, using default font.")

    # --- 4. Loop Processing ---
    for i, p in enumerate(sampled):
        print(f"[{i+1}/{num_samples}] Processing {p.name}...")

        # YOLO Inference
        results = model(str(p), imgsz=1280, verbose=False)
        r = results[0]

        # 4.1 Generating the original image with bounding boxes (YOLO plot)
        # Dynamically adjust line width and font size
        lw = max(2, r.orig_shape[1] // 200)
        im_pil = r.plot(pil=True, line_width=lw, font_size=max(12, r.orig_shape[1] // 50))

        # 4.2 Retrieving Decision Rule Text
        detected_indices = r.boxes.cls.cpu().numpy().astype(int)
        detected_names = {model.names[idx] for idx in detected_indices}

        rules_to_show = []
        for name in detected_names:
            if name in DECISION_RULES:
                rules_to_show.append(f"{name}: {DECISION_RULES[name]}")

        if not rules_to_show:
            rules_to_show = ["Safe driving: No specific signs detected."]

        # 4.3 Dynamic Layout Calculation (Right Panel Logic)
        w_orig, h_orig = im_pil.size

        # Font size set to 1/35 of the original image width (Reactive)
        font_size = max(20, int(w_orig / 35))
        line_spacing = int(font_size * 1.5)
        padding = int(font_size * 0.8)

        # Reload fonts of a specific size
        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except (IOError, TypeError):
            font = ImageFont.load_default()

        # 1. Calculate the maximum width required for the whiteboard on the right
        draw_temp = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        max_text_width = 0
        for rule in rules_to_show:
            try:
                text_bbox = draw_temp.textbbox((0, 0), rule, font=font)
                text_width = text_bbox[2] - text_bbox[0]
            except AttributeError:
                text_width = draw_temp.textsize(rule, font=font)[0]

            max_text_width = max(max_text_width, text_width)

        # Panel width = Maximum text width + Left and right margins
        panel_width = max_text_width + padding * 2

        # 2. Calculate the final canvas dimensions
        final_w = w_orig + panel_width
        final_h = h_orig

        # 4.4 Creating a New Canvas (Right-hand panel)
        final_img = Image.new('RGB', (final_w, final_h), color='white')

        # Paste the original image to the left (at coordinates x=0, y=0)
        final_img.paste(im_pil, (0, 0))

        # 4.5 Drawing Text (centred within the right-hand panel)
        draw = ImageDraw.Draw(final_img)

        # Calculate the total text height (for vertical centring)
        total_text_height = len(rules_to_show) * line_spacing - (line_spacing - font_size)

        # Vertical starting coordinate (vertically centred)
        current_y = (final_h - total_text_height) / 2
        current_y = max(padding, current_y)


        # Draw text line by line (left-aligned within the right-hand panel)
        for rule in rules_to_show:
            # Text start X-coordinate = Original image width + Left padding
            start_x = w_orig + padding

            draw.text(
                (start_x, current_y),
                rule,
                fill="black",
                font=font
            )
            current_y += line_spacing

        # 4.6 Saving Images
        out_file = OUT_DIR / f"pred_{p.name}"
        final_img.save(out_file)

        # 4.7 Pop-up Display (Display image in Notebook)
        print(f"Displaying {p.name} with right panel...")
        display(final_img)
        # To prevent continuous output from causing the notebook to crash, pause for 1 second.
        time.sleep(1)


    print(f"\nDone! All {num_samples} images saved to: {OUT_DIR.absolute()}")

# -----------------
# Execute the main function
# -----------------
main()