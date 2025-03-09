import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#  Load Final Model
model = load_model("pokemon_cnn_final.h5")

# Load Class Labels
try:
    with open("pokemon_mapping.json", "r") as f:
        class_indices = json.load(f)
        class_labels = {v: k for k, v in class_indices.items()}
except FileNotFoundError:
    print("Error: pokemon_mapping.json not found!")
    class_labels = {}

# Prediction Function
def predict_image(image_path):
    test_image = image.load_img(image_path, target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)
    class_index = np.argmax(result)

    if class_index in class_labels:
        return f'{class_labels[class_index]} ({result[0][class_index] * 100:.2f}%)'
    else:
        return "Unknown Pokémon (Invalid Prediction)"

#  GUI Implementation
def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img
    prediction = predict_image(file_path)
    result_label.config(text=prediction)

#  GUI Window
root = tk.Tk()
root.title("Pokémon Classifier")
root.geometry("400x500")

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack()

root.mainloop()

