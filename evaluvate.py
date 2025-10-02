import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("sinhala_cnn_model.h5")

# Parameters
img_height, img_width = 64, 64

# Folder with test images
test_folder = "test_set"

# Get all image files in subfolders
image_files = []
for root, dirs, files in os.walk(test_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

# Reconstruct class names from training folder
train_folder = "dataset/"  # same folder used for training
class_names = sorted([d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))])
print("Class names:", class_names)

# Display first 9 test images with predictions
plt.figure(figsize=(12, 12))
for i, img_path in enumerate(image_files[:9]):
    img = image.load_img(img_path, target_size=(img_height, img_width), color_mode="grayscale")
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    predicted_label = class_names[np.argmax(pred)]

    ax = plt.subplot(3, 3, i+1)
    plt.imshow(x[0].squeeze(), cmap="gray")
    plt.title(f"Pred: {predicted_label}")
    plt.axis("off")

# Save the figure as PNG
plt.savefig("sinhala_predictions.png")
print("Saved visual predictions as sinhala_predictions.png")
