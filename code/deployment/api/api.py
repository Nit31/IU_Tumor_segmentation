from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import torch
import numpy as np
import unet
import base64

app = FastAPI()

# Load the trained U-Net model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = unet.UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('model/model.pth', map_location=device))
model.to(device)
model.eval()

def preprocess_image(image):
    target_size = (128, 128)

    # Preprocess the uploaded image
    image = cv2.resize(image, target_size)
    image_tensor = torch.Tensor(image) / 255
    return image_tensor.unsqueeze(0).unsqueeze(0)

def postprocess_mask(mask: torch.Tensor):
    # Postprocess the mask tensor to return as a response
    mask = mask.squeeze(0).cpu().numpy().squeeze() > 0.5
    mask = mask.astype(np.float32)
    mask = cv2.resize(mask, (640, 640))
    mask *= 255
    return mask

def overlay_mask(image, mask):
    # Create a copy of the image with 3 channels
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Ensure the mask is in the correct dtype
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, image.shape[:2])

    # Apply the mask to the overlay (set color in overlay)
    overlay[:, :, 1] += (mask // 3)  # Add a portion of the mask value to the green channel
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)  # Ensure values stay in uint8 range
    print(overlay.shape)
    return overlay


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load the uploaded image
    image_data = await file.read()
    # image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Ensure 3-channel RGB
    image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    image_tensor = preprocess_image(image)
    # Run the model inference
    with torch.no_grad():
        output_mask = model(image_tensor.to(device))
    print(output_mask.max())
    # Postprocess the mask
    processed_mask = postprocess_mask(output_mask)

    # Overlay the mask on the original image
    output_mask_image = overlay_mask(image, processed_mask)

    # Convert the output image to Base64
    _, buffer = cv2.imencode('.jpg', output_mask_image)  # Encode image to jpg
    base64_image = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string


    return JSONResponse(content={
        "message": "Prediction complete!",
        "overlay": base64_image
    })