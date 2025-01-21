from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import cv2
import io
import numpy as np
import base64
import traceback
import os

app = FastAPI()

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Mount the static directory to serve files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model
try:
    model = tf.keras.models.load_model("D:/Retina Blood Vessel Segmentation/unet_model.h5")
except Exception as e:
    print(f"Error loading the model: {str(e)}")
    model = None


def preprocess_image(image: Image.Image):
    """
    Preprocess image: resize, convert to grayscale, equalize, and detect edges.
    """
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(image) / 255.0
    gray_img = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    equalized_img = cv2.equalizeHist(gray_img) / 255.0
    return np.expand_dims(equalized_img, axis=-1)


@app.get("/", response_class=HTMLResponse)
async def get_form():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Retina Blood Vessel Segmentation</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f5f5f5;
                margin: 0;
                padding: 0;
                text-align: center;
            }
            header {
                background-color: #0078d7;
                color: white;
                padding: 20px;
                font-size: 1.5rem;
            }
            form {
                background-color: white;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                display: inline-block;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
            }
            input, button {
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin: 10px 0;
                width: 80%;
                font-size: 1rem;
            }
            button {
                background-color: #0078d7;
                color: white;
                cursor: pointer;
                border: none;
            }
            button:hover {
                background-color: #005fa3;
            }
            #result {
                margin-top: 20px;
            }
            img {
                max-width: 100%;
                height: auto;
                margin-top: 10px;
                border: 2px solid #ccc;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <header>Retina Blood Vessel Segmentation</header>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <label for="file">Upload an Image:</label><br>
            <input type="file" name="file" id="file" accept="image/*" required><br>
            <button type="submit">Predict</button>
        </form>
        <div id="result">
            <h2>Prediction Result:</h2>
            <div id="images"></div>
        </div>
        <script>
            const form = document.querySelector("form");
            form.addEventListener("submit", async function (event) {
                event.preventDefault();
                const formData = new FormData(form);
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData,
                });
                const result = await response.json();
                const resultDiv = document.getElementById("images");
                resultDiv.innerHTML = `
                    <h3>Original Image:</h3>
                    <img src="data:image/png;base64,${result.original_image}" alt="Original Image">
                    <h3>Prediction Mask:</h3>
                    <img src="data:image/png;base64,${result.prediction_image}" alt="Prediction Mask">
                `;
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict", response_class=JSONResponse)
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded correctly.")

    try:
        # Read image and preprocess
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Resize original image to 128x128 before encoding it for the result
        resized_image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        
        processed_image = preprocess_image(image)
        image_input = np.expand_dims(processed_image, axis=0)

        # Make prediction
        pred = model.predict(image_input)
        pred_mask = (pred > 0.5).astype(np.uint8)
        pred_img = Image.fromarray(pred_mask[0, :, :, 0] * 255)

        # Encode original (resized) and prediction images
        buffer_original = io.BytesIO()
        resized_image.save(buffer_original, format="PNG")
        encoded_original = base64.b64encode(buffer_original.getvalue()).decode("utf-8")

        buffer_pred = io.BytesIO()
        pred_img.save(buffer_pred, format="PNG")
        encoded_pred = base64.b64encode(buffer_pred.getvalue()).decode("utf-8")

        return JSONResponse(content={"original_image": encoded_original, "prediction_image": encoded_pred})

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
