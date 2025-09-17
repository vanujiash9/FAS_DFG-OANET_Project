import os
import cv2
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
import json
from collections import Counter, defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg
from src.visualization.viz import plot_data_distribution, plot_split_distribution

detector = MTCNN()

def align_and_crop_face(image_path, target_size, margin_ratio):
    img = cv2.imread(image_path)
    if img is None: return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img_rgb)

    if not detections: return None

    detection = max(detections, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = detection['box']

    margin_x = w * margin_ratio
    margin_y = h * margin_ratio
    x = max(int(x - margin_x / 2), 0)
    y = max(int(y - margin_y / 2), 0)
    w = int(w + margin_x)
    h = int(h + margin_y)

    face = img[y:y+h, x:x+w]
    if face.size == 0: return None
    face = cv2.resize(face, target_size)
    return face

def process_and_save_images_from_dir(input_dir, output_dir, label_prefix, target_size, margin_ratio, max_images=None, ignore_existing=False):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if max_images:
        image_files = image_files[:max_images]

    processed_count = 0
    for img_file in tqdm(image_files, desc=f"processing {label_prefix} from {input_dir}"):
        output_filename = f"{label_prefix}_{os.path.splitext(img_file)[0]}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        if ignore_existing and os.path.exists(output_path):
            processed_count += 1
            continue

        img_path = os.path.join(input_dir, img_file)
        aligned_face = align_and_crop_face(img_path, target_size, margin_ratio)
        if aligned_face is not None:
            cv2.imwrite(output_path, aligned_face)
            processed_count += 1
    return processed_count


if __name__ == "__main__":
    print("--- preprocessing dfg live data ---")
    dfg_live_count = 0
    dfg_live_count += process_and_save_images_from_dir(
        cfg.data.CELEBA_SPOOF_LIVE_FRAMES_DIR, cfg.data.DFG_REAL_FACES_TRAIN_DIR,
        "live_celeba", (cfg.data.DFG_IMAGE_SIZE, cfg.data.DFG_IMAGE_SIZE),
        cfg.data.FACE_ALIGNMENT_MARGIN_RATIO
    )
    dfg_live_count += process_and_save_images_from_dir(
        cfg.data.FFHQ_RAW_DIR, cfg.data.DFG_REAL_FACES_TRAIN_DIR,
        "live_ffhq", (cfg.data.DFG_IMAGE_SIZE, cfg.data.DFG_IMAGE_SIZE),
        cfg.data.FACE_ALIGNMENT_MARGIN_RATIO
    )
    print(f"\nTotal dfg live images processed: {dfg_live_count}")

    print("\n--- preprocessing oanet data (live & spoof) ---")
    oa_net_live_count = 0
    oa_net_spoof_count = 0
    spoof_types_distribution = Counter()

    oa_net_live_output_dir = os.path.join(cfg.data.PROCESSED_DATA_DIR, 'oanet_dataset', 'live')
    oa_net_live_count += process_and_save_images_from_dir(
        cfg.data.CELEBA_SPOOF_LIVE_FRAMES_DIR, oa_net_live_output_dir,
        "live", (cfg.data.OANET_IMAGE_SIZE, cfg.data.OANET_IMAGE_SIZE),
        cfg.data.FACE_ALIGNMENT_MARGIN_RATIO
    )

    spoof_labels_map = {}
    if os.path.exists(cfg.data.CELEBA_SPOOF_LABELS_PATH):
        try:
            with open(cfg.data.CELEBA_SPOOF_LABELS_PATH, 'r') as f:
                full_labels_data = json.load(f)
                for img_name, data in full_labels_data.items():
                    if data.get("label") == 1:
                        spoof_labels_map[img_name] = data.get("attack_type", "unknown")
        except Exception as e:
            print(f"warning: could not load or parse {cfg.data.CELEBA_SPOOF_LABELS_PATH}: {e}")
    
    spoof_files_by_type = defaultdict(list)
    all_raw_spoof_files = [f for f in os.listdir(cfg.data.CELEBA_SPOOF_SPOOF_FRAMES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in all_raw_spoof_files:
        original_img_name = os.path.splitext(img_file)[0]
        attack_type = spoof_labels_map.get(original_img_name, "unknown")
        spoof_files_by_type[attack_type].append(img_file)

    print(f"processing {len(all_raw_spoof_files)} spoof images from {cfg.data.CELEBA_SPOOF_SPOOF_FRAMES_DIR}...")
    for attack_type, files_in_type in spoof_files_by_type.items():
        type_output_dir = os.path.join(cfg.data.PROCESSED_DATA_DIR, 'oanet_dataset', 'spoof', attack_type)
        os.makedirs(type_output_dir, exist_ok=True)
        
        for img_file in tqdm(files_in_type, desc=f"processing '{attack_type}' spoofs"):
            output_filename = f"spoof_{os.path.splitext(img_file)[0]}.jpg"
            output_path = os.path.join(type_output_dir, output_filename)

            if os.path.exists(output_path):
                oa_net_spoof_count += 1
                spoof_types_distribution[attack_type] += 1
                continue

            img_path = os.path.join(cfg.data.CELEBA_SPOOF_SPOOF_FRAMES_DIR, img_file)
            aligned_face = align_and_crop_face(
                img_path, (cfg.data.OANET_IMAGE_SIZE, cfg.data.OANET_IMAGE_SIZE),
                cfg.data.FACE_ALIGNMENT_MARGIN_RATIO
            )
            if aligned_face is not None:
                cv2.imwrite(output_path, aligned_face)
                oa_net_spoof_count += 1
                spoof_types_distribution[attack_type] += 1


    print(f"\nTotal oanet live images processed: {oa_net_live_count}")
    print(f"Total oanet spoof images processed: {oa_net_spoof_count}")

    print("\n--- visualizing data distribution ---")
    plot_data_distribution(dfg_live_count, oa_net_live_count, oa_net_spoof_count, spoof_types_distribution)

    print("\n--- splitting oanet data into train/val/test ---")
    all_oa_net_images = []
    all_oa_net_labels = []

    live_processed_dir = os.path.join(cfg.data.PROCESSED_DATA_DIR, 'oanet_dataset', 'live')
    live_files = [os.path.join(live_processed_dir, f) for f in os.listdir(live_processed_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_oa_net_images.extend(live_files)
    all_oa_net_labels.extend([0] * len(live_files))

    spoof_processed_root = os.path.join(cfg.data.PROCESSED_DATA_DIR, 'oanet_dataset', 'spoof')
    for spoof_type_dir in os.listdir(spoof_processed_root):
        type_path = os.path.join(spoof_processed_root, spoof_type_dir)
        if os.path.isdir(type_path):
            spoof_files_type = [os.path.join(type_path, f) for f in os.listdir(type_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_oa_net_images.extend(spoof_files_type)
            all_oa_net_labels.extend([1] * len(spoof_files_type))

    if not all_oa_net_images:
        print("no images found for oanet split.")
    else:
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            all_oa_net_images, all_oa_net_labels, test_size=0.2, stratify=all_oa_net_labels, random_state=42
        )
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )

        split_data = {
            'train': [{'path': f, 'label': l} for f, l in zip(train_files, train_labels)],
            'val': [{'path': f, 'label': l} for f, l in zip(val_files, val_labels)],
            'test': [{'path': f, 'label': l} for f, l in zip(test_files, test_labels)],
        }
        with open(cfg.data.OANET_SPLITS_PATH, 'w') as f:
            json.dump(split_data, f, indent=4)

        print(f"oanet data split: train {len(train_files)} | val {len(val_files)} | test {len(test_files)}")
        plot_split_distribution(len(train_files), len(val_files), len(test_files))
    print("Data preprocessing and splitting complete.")