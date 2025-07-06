import cv2
from flask import Flask, jsonify, request, render_template
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, InpaintRequest, LDMSampler
from PIL import Image
import numpy as np
import torch
import io
import base64


# --- AI Model Initialization ---
# Initialize the LaMa model for inpainting.
# This is done once when the app starts.
# It checks for a GPU (cuda) and falls back to CPU if not available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Create a comprehensive Config object with all required fields
# High-quality configuration to achieve a more seamless and natural result.
# - HDStrategy.CROP: Splits the image into parts for high-resolution processing.
# - ldm_steps=50: Increases the number of generation steps for finer detail.
lama_config = InpaintRequest(
    hd_strategy=HDStrategy.ORIGINAL,
    ldm_sampler=LDMSampler.ddim,
    ldm_steps=50
)
model = ModelManager(name="lama", device=device, config=lama_config)
# -----------------------------

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def resize_with_padding(image: Image.Image, max_dim: int, padding_ratio: float = 0.05):
    """
    画像をアスペクト比を維持したままリサイズし、指定された最大辺の長さに収まるように調整し、
    さらに白い余白を追加します。

    Args:
        image (PIL.Image.Image): 処理する画像。
        max_dim (int): 画像の長い方の辺の最大ピクセル数。
        padding_ratio (float): 余白の比率。画像の一番長い辺に対する余白の割合。

    Returns:
        PIL.Image.Image: リサイズされ、余白が追加された画像。
    """
    original_width, original_height = image.size
    
    # アスペクト比を維持しつつ、max_dimに収まるようにリサイズ
    if original_width > original_height:
        new_width = max_dim
        new_height = int(original_height * (max_dim / original_width))
    else:
        new_height = max_dim
        new_width = int(original_width * (max_dim / original_height))

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # 余白の計算
    # 最も長い辺を基準に余白を計算
    base_dim = max(new_width, new_height)
    padding_size = int(base_dim * padding_ratio)

    # 新しいキャンバスサイズ
    padded_width = new_width + 2 * padding_size
    padded_height = new_height + 2 * padding_size

    # 白い背景の新しい画像を作成
    padded_image = Image.new("RGB", (padded_width, padded_height), (255, 255, 255))
    
    # リサイズされた画像を中央に配置
    paste_x = padding_size
    paste_y = padding_size
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image


@app.route('/erase', methods=['POST'])
def erase():
    try:
        image_file = request.files.get('image')
        mask_file = request.files.get('mask')

        if not image_file or not mask_file:
            return jsonify({"error": "画像またはマスクのデータがありません"}), 400

        # Open images
        image_data = Image.open(image_file.stream).convert('RGB')
        mask_data = Image.open(mask_file.stream).convert('RGB')

        # Define a maximum dimension for processing (e.g., 512x512)
        # This will be the longer side of the image after resizing, before padding.
        max_dim_for_processing = 512 

        # アスペクト比を維持してリサイズ（余白なし）
        original_width, original_height = image_data.size
        if original_width > original_height:
            new_width = max_dim_for_processing
            new_height = int(original_height * (max_dim_for_processing / original_width))
        else:
            new_height = max_dim_for_processing
            new_width = int(original_width * (max_dim_for_processing / original_height))

        image_data = image_data.resize((new_width, new_height), Image.LANCZOS)
        # マスクも同じサイズにリサイズ
        mask_data = mask_data.resize((new_width, new_height), Image.LANCZOS)

        # Pillow画像を明示的にuint8型のNumPy配列へ変換
        image_np = np.array(image_data)
        if image_np.dtype == np.float64 or image_np.dtype == np.float32:
            image_np = np.clip(image_np * 255 if image_np.max() <= 1.0 else image_np, 0, 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        mask_np = np.array(mask_data)
        # If the mask is grayscale, convert it to 3 channels
        if mask_np.ndim == 2:
            mask_np = np.stack([mask_np, mask_np, mask_np], axis=-1)

        if mask_np.dtype == np.float64 or mask_np.dtype == np.float32:
            mask_np = np.clip(mask_np * 255 if mask_np.max() <= 1.0 else mask_np, 0, 255).astype(np.uint8)
        else:
            mask_np = mask_np.astype(np.uint8)

        # Create a robust binary mask from the red lines on the mask image
        # We define "red" as pixels with a high R value and low G and B values.
        is_red_mask = (mask_np[:, :, 0] > 200) & (mask_np[:, :, 1] < 100) & (mask_np[:, :, 2] < 100)
        binary_mask = np.where(is_red_mask, 255, 0).astype(np.uint8)
        
        # Perform high-quality inpainting using the LaMa model (which expects BGR)
        # Only run the model if there is something to inpaint.
        if np.sum(binary_mask) > 0:
            # Ensure binary_mask is 2D (H, W) and uint8 before passing to the model
            binary_mask = np.squeeze(binary_mask) # Remove all singleton dimensions
            if binary_mask.ndim != 2:
                # If it's still not 2D, force it to 2D (H, W)
                binary_mask = binary_mask.reshape(binary_mask.shape[0], binary_mask.shape[1])
            binary_mask = binary_mask.astype(np.uint8) # Ensure dtype

            # Ensure mask and image have the same dimensions
            if image_bgr.shape[:2] != binary_mask.shape[:2]:
                print(f"Resizing mask from {binary_mask.shape[:2]} to {image_bgr.shape[:2]}")
                binary_mask = cv2.resize(binary_mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            print(f"Image shape before model: {image_bgr.shape}")
            print(f"Mask shape before model: {binary_mask.shape}")

            result_bgr = model(image_bgr, binary_mask, config=lama_config)
            
            # Convert model output to a displayable format.
            # The model outputs float64 in [0, 255], which needs to be uint8.
            # We clip the values to ensure they are within the valid 0-255 range
            # before converting the data type.
            result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)

            if result_bgr.ndim == 2: # Grayscale (H, W)
                result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_GRAY2BGR)
            elif result_bgr.ndim == 3 and result_bgr.shape[2] == 1: # Grayscale (H, W, 1)
                result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_GRAY2BGR)
            elif result_bgr.ndim == 3 and result_bgr.shape[2] == 4: # BGRA (H, W, 4)
                result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_BGRA2BGR)
        else:
            # If no mask is drawn, just use the original image
            result_bgr = image_bgr

        # The model appears to output in RGB format directly.
        # Therefore, we will use its output as is, without BGR to RGB conversion.
        result_rgb = result_bgr

        # Convert the final result back to a Pillow image
        result_image = Image.fromarray(result_rgb)
        
        # Save the result to an in-memory buffer
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        
        # Encode the image to a base64 string to send back to the client
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return the base64 string in a JSON response
        return jsonify({"image": "data:image/png;base64," + img_str})

    except Exception as e:
        # Log the full error to the server console for debugging
        print(f"An error occurred: {e}")
        # Return a generic error message to the client
        return jsonify({"error": "サーバー内部でエラーが発生しました。"}), 500

if __name__ == "__main__":
    app.run()