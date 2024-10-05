import torch
from torchvision import transforms


from PIL import Image

data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64 (width x height)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = torch.load("./model/simple_model")
labels = ['cat','fish']
img = Image.open("pexels-crisdip-35358-128756.jpg")
# img = Image.open("./images/val/coho.jpg")
img = data_transforms(img)
img = img.unsqueeze(0)
prediction = model(img)
print(prediction)
prediction = prediction.argmax()
print(labels[prediction])