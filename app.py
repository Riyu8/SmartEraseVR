from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import io
import base64
import cv2
import torch
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler

# --- AI Model Initialization ---
# Initialize the LaMa model for inpainting.
# This is done once when the app starts.
# It checks for a GPU (cuda) and falls back to CPU if not available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Create a comprehensive Config object with all required fields
lama_config = Config(
    hd_strategy=HDStrategy.ORIGINAL,
    ldm_sampler=LDMSampler.ddim,
    ldm_steps=20,                      # Sampling steps
    hd_strategy_crop_margin=32,        # Margin for cropping (pixels)
    hd_strategy_crop_trigger_size=512, # Size triggering HD strategy
    hd_strategy_resize_limit=2048      # Maximum image size limit
)
model = ModelManager(name="lama", device=device, config=lama_config)
# -----------------------------

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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
