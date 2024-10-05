from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms

# Load the pre-trained model (ensure that the path is correct)
model = torch.load("./model/simple_model")
model.eval()  # Set the model to evaluation mode

# Define class labels
labels = ['cat', 'fish']

# Define the image transformation (must match the transformation used during training)
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create FastAPI instance
app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Load the uploaded image
        img = Image.open(file.file)
        
        # Apply transformations to the image
        img = data_transforms(img)
        img = img.unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():  # Disable gradient computation
            output = model(img)
            prediction = output.argmax().item()

        # Return the result
        return JSONResponse(content={"label": labels[prediction]})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

