# File: src/scripts/predict.py
# CÔNG CỤ DỰ ĐOÁN TƯƠNG TÁC CHO ẢNH VÀ VIDEO

import os
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm
import argparse
from torchvision import transforms

# --- Cấu hình hệ thống ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.configs.config_loader import cfg
from src.data_management.datasets import get_oanet_eval_transforms
from src.models.oanet.network import OANet
from src.models.dfg.generator import DFGGenerator

# ... (Toàn bộ các hàm process_image, process_video giữ nguyên như trước)
def predict(input_path, output_dir):
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    print("Đang tải các mô hình...")
    oanet_model_path = os.path.join(cfg.oanet.OANET_CHECKPOINTS_DIR, "best_oanet_vit_model.pth")
    oanet_model = OANet(device=device)
    state_dict = torch.load(oanet_model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    oanet_model.load_state_dict(state_dict)
    oanet_model.to(device)
    oanet_model.eval()
    dfg_unet_path = os.path.join(cfg.dfg.DFG_CHECKPOINTS_DIR, "best_dfg_unet.pth")
    dfg_model = DFGGenerator(device=device)
    dfg_model.unet.load_state_dict(torch.load(dfg_unet_path, map_location=device))
    dfg_model.to(device)
    dfg_model.eval()
    print("Tải mô hình thành công.")
    transform_func = get_oanet_eval_transforms()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_extension = Path(input_path).suffix.lower()
    if file_extension in ['.jpg', '.jpeg', '.png']:
        process_image(input_path, oanet_model, dfg_model, transform_func, device, output_path)
    elif file_extension in ['.mp4', '.avi', '.mov']:
        process_video(input_path, oanet_model, dfg_model, transform_func, device, output_path)
    else:
        print(f"Lỗi: Định dạng file '{file_extension}' không được hỗ trợ.")

def process_image(img_path, oanet_model, dfg_model, transform, device, output_dir):
    print(f"Đang xử lý ảnh: {img_path}")
    image_pil = Image.open(img_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        _, cue_tensor = dfg_model.generate_reconstruction_and_cue(image_tensor)
        logits = oanet_model(cue_tensor)
        prob = torch.sigmoid(logits).squeeze().item()
        prediction = "SPOOF" if prob > cfg.oanet.OANET_EVAL_THRESHOLD else "REAL"
        cue_tensor_norm = (cue_tensor.squeeze(0) - cue_tensor.min()) / (cue_tensor.max() - cue_tensor.min())
        cue_img = transforms.ToPILImage()(cue_tensor_norm.cpu())
    result_img = image_pil.copy()
    draw = ImageDraw.Draw(result_img)
    try: font = ImageFont.truetype("arial.ttf", 40)
    except IOError: font = ImageFont.load_default()
    text = f"Prediction: {prediction} (Score: {prob:.2f})"
    color = (255, 0, 0) if prediction == "SPOOF" else (0, 255, 0)
    draw.text((10, 10), text, font=font, fill=color)
    final_output = Image.new('RGB', (result_img.width + cue_img.width, result_img.height))
    final_output.paste(result_img, (0, 0))
    final_output.paste(cue_img, (result_img.width, 0))
    output_filename = f"predicted_{Path(img_path).name}"
    output_filepath = output_dir / output_filename
    final_output.save(output_filepath)
    print(f"Kết quả dự đoán đã được lưu tại: {output_filepath}")

def process_video(video_path, oanet_model, dfg_model, transform, device, output_dir):
    print(f"Đang xử lý video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)); frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_filename = f"predicted_{Path(video_path).name}"
    output_filepath = str(output_dir / output_filename)
    out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    with torch.no_grad():
        for _ in tqdm(range(frame_count), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            image_tensor = transform(frame_pil).unsqueeze(0).to(device)
            _, cue_tensor = dfg_model.generate_reconstruction_and_cue(image_tensor)
            logits = oanet_model(cue_tensor)
            prob = torch.sigmoid(logits).squeeze().item()
            prediction = "SPOOF" if prob > cfg.oanet.OANET_EVAL_THRESHOLD else "REAL"
            text = f"{prediction} (Score: {prob:.2f})"
            color = (0, 0, 255) if prediction == "SPOOF" else (0, 255, 0)
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            out.write(frame)
    cap.release()
    out.release()
    print(f"Video đã xử lý và được lưu tại: {output_filepath}")

# ==============================================================================
# PHẦN THỰC THI CHÍNH (ĐÃ NÂNG CẤP)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán Real/Spoof cho một ảnh hoặc video.")
    # Thêm `nargs='?'` để `input_path` trở thành tùy chọn (optional)
    parser.add_argument("input_path", type=str, nargs='?', default=None,
                        help="Đường dẫn đến file ảnh/video. Nếu bỏ trống, sẽ vào chế độ tương tác.")
    
    default_output_dir = os.path.join(cfg.oanet.RESULTS_DIR, "predictions")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help="Thư mục để lưu kết quả.")
    
    args = parser.parse_args()
    
    target_file = None

    if args.input_path:
        # Chế độ 1: Người dùng cung cấp đường dẫn trực tiếp
        target_file = args.input_path
    else:
        # Chế độ 2: Tương tác - Chọn file từ thư mục `uploads`
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True) # Tự động tạo thư mục nếu chưa có
        
        print("\n" + "="*50)
        print("   CHẾ ĐỘ DỰ ĐOÁN TƯƠNG TÁC")
        print("="*50)
        
        # Liệt kê các file có sẵn
        supported_extensions = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
        files_in_uploads = [f for f in uploads_dir.iterdir() if f.suffix.lower() in supported_extensions]
        
        if not files_in_uploads:
            print(f"\n!!! Thư mục '{uploads_dir}' đang trống.")
            print("HƯỚNG DẪN:")
            print("1. Mở trình quản lý file của bạn (ví dụ: File Explorer trên Windows).")
            print(f"2. Kéo và thả file ảnh/video bạn muốn dự đoán vào thư mục '{uploads_dir}' trên server.")
            print("   (Bạn có thể dùng tính năng kéo-thả của VS Code hoặc một công cụ SFTP như WinSCP, FileZilla).")
            print("3. Chạy lại script này.")
            sys.exit(0)

        print("\nCác file có sẵn trong thư mục 'uploads':")
        for i, file_path in enumerate(files_in_uploads):
            print(f"  [{i+1}] {file_path.name}")
        
        # Hỏi người dùng chọn file
        while True:
            try:
                choice = int(input("\n> Vui lòng chọn số thứ tự của file bạn muốn xử lý: "))
                if 1 <= choice <= len(files_in_uploads):
                    target_file = files_in_uploads[choice - 1]
                    break
                else:
                    print(f"Lựa chọn không hợp lệ. Vui lòng chọn một số từ 1 đến {len(files_in_uploads)}.")
            except ValueError:
                print("Lựa chọn không hợp lệ. Vui lòng nhập một con số.")

    # --- Bắt đầu dự đoán ---
    if target_file and os.path.exists(target_file):
        predict(str(target_file), args.output_dir)
    else:
        print(f"Lỗi: File '{target_file}' không tồn tại.")