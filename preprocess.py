from tvm.contrib.download import download_testdata
from PIL import Image, ImageOps
import numpy as np
import cv2
import urllib.request
img_url = "https://drive.google.com/uc?export=download&id=1eNeyI82KaU3UnF3kTl5bO6DlmCWfxeJJ"
img_path = download_testdata(img_url, "imagenet_cat1.png", module="data")


imgURL = "https://drive.google.com/uc?export=download&id=1eNeyI82KaU3UnF3kTl5bO6DlmCWfxeJJ"

urllib.request.urlretrieve(imgURL, "a.png")

image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (28,28)).astype(np.float32)/255
input = np.reshape(gray, (1,1,28,28))
print(input.shape)
np.savez("imagenet_num1", input=input)
# # Resize it to 224x224
# resized_image = Image.open(img_path).resize((28, 28))
# resized_image = ImageOps.grayscale(resized_image)

# img_data = np.asarray(resized_image).astype("float32")
# print(img_data.shape,"is shape")

# img_data = np.expand_dims(img_data, axis=0)
# print(img_data.shape,"is shape")
# # # ONNX expects NCHW input, so convert the array
# # img_data = np.transpose(img_data, (2, 0, 1))

# # Normalize according to ImageNet
# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_stddev = np.array([0.229, 0.224, 0.225])
# norm_img_data = np.zeros(img_data.shape).astype("float32")


# # # # Add batch dimension
# img_data = np.expand_dims(img_data, axis=0)
# print(img_data.shape,"is shape")
# Save to .npz (outputs imagenet_cat.npz)

# 1. Check if image is same or not
# 2. 