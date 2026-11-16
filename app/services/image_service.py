import io
import base64
import numpy as np
from PIL import Image

def preprocess_image_for_model(image_bytes_or_stream):
    """
    Exact preprocessing matching Notebook Cells 43, 44 & 48:
    1. Load image and convert to RGB
    2. Resize to (32, 32)
    3. Convert to numpy array
    4. Normalize pixel values by dividing by 255.0
    5. Expand dims to shape (1, 32, 32, 3)

    Returns:
        np.ndarray with shape (1, 32, 32, 3) and float32 dtype in [0.0, 1.0]
    """
    if isinstance(image_bytes_or_stream, (bytes, bytearray)):
        stream = io.BytesIO(image_bytes_or_stream)
    else:
        stream = image_bytes_or_stream

    img = Image.open(stream)
    img_rgb = img.convert("RGB")
    img_32x32 = img_rgb.resize((32, 32), Image.Resampling.LANCZOS)

    arr = np.array(img_32x32, dtype=np.float32) / 255.0
    arr_batch = np.expand_dims(arr, axis=0)
    return arr_batch, img_rgb


def create_data_uri(image_pil, max_size=(255, 255)):
    """Generate Base64 data URI for instant web display."""
    preview = image_pil.copy()
    preview.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=90)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"
