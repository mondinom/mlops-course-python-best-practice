import os
from typing import List

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models.resnet import ResNet18_Weights


class ImageData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_images(self) -> List[Image.Image]:
        images = []
        for filename in os.listdir(self.data_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                images.append(Image.open(os.path.join(self.data_path, filename)))
        return images


class ImageProcess:
    def __init__(self, size: int):
        self.size = size

    def resize_and_gray(self, img_list: List[Image.Image]) -> List[torch.Tensor]:
        """
        Resize, normalize and convert PIL images to grayscale tensor. (reST style Docstrings, for eg)

        :param img_list: List of PIL Image objects.
        :type img_list: List[Image.Image]
        :returns: List of processed image tensors.
        :rtype: List[torch.Tensor]
        """
        processed_images = []
        transform = transforms.Compose(
            [
                transforms.Resize((self.size, self.size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        for img in img_list:
            processed_images.append(transform(img))
        return processed_images


class Predictor:
    def __init__(self):
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def predict_img(self, processed_images: List[torch.Tensor]):
        """
        Predicts classes for a list of processed image tensors. (Google style Docstrings, for eg)

        Args:
            processed_images (List[torch.Tensor]): List of processed image tensors.

        Returns:
            List[int]: List of predicted class indices.
        """
        results = []
        for img_tensor in processed_images:
            pred = self.mdl(img_tensor.unsqueeze(0))
            results.append(torch.argmax(pred, dim=1).item())
        return results


if __name__ == "__main__":
    loader = ImageData("images/")
    images = loader.load_images()

    processor = ImageProcess(256)
    processed_images = processor.resize_and_gray(images)

    pred = Predictor()
    results = pred.predict_img(processed_images)
    print(results)
