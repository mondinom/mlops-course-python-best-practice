import os
import sys
from PIL import Image
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from image_processing import ImageData, ImageProcess, Predictor  # noqa: E402


# def test_load_images():
#     loader = ImageData("tests_images/")
#     images = loader.load_images()
#     assert isinstance(images, list)
#     if images:
#         assert isinstance(images[0], Image.Image)


def test_resize_and_gray():
    processor = ImageProcess(256)
    test_image = Image.new("RGB", (512, 512))
    processed_images = processor.resize_and_gray([test_image])
    assert len(processed_images) == 1
    assert isinstance(processed_images[0], torch.Tensor)
    assert processed_images[0].shape == torch.Size([3, 256, 256])


def test_predict_img():
    predictor = Predictor()
    test_tensor = torch.randn(3, 256, 256)  # Random tensor simulating processed image
    results = predictor.predict_img([test_tensor])
    assert isinstance(results, list)
    assert isinstance(results[0], int)
