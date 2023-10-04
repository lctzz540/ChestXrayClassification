import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from net import CNNnet


def load_model(model_path):
    model = CNNnet()
    model.load_state_dict(torch.load(model_path))
    return model


def predict_single_image(model, image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    predicted_class = "PNEUMONIA" if output.item() > 0.5 else "NORMAL"

    return predicted_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict pneumonia in a chest X-ray image."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("model_path", type=str, help="Path to the trained model.")

    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print("Error: The specified image file does not exist.")
    elif not os.path.isfile(args.model_path):
        print("Error: The specified model file does not exist.")
    else:
        model = load_model(args.model_path)
        predicted_class = predict_single_image(model, args.image_path)
        print(f"Predicted class: {predicted_class}")
