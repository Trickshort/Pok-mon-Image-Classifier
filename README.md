Pokémon Image Classifier

📌 Project Overview
The Pokémon Image Classifier is a deep learning-based project that can identify Pokémon from images using a Convolutional Neural Network (CNN). The model is trained on a dataset containing various Pokémon species, allowing it to classify an input image into one of the known Pokémon categories.

This project consists of:
✅ Model Training (train.py) – Trains a CNN on a Pokémon dataset.
✅ Model Testing (test.py) – Loads the trained model and predicts Pokémon from images.
✅ Graphical User Interface (pokemon_gui.py) – Provides a user-friendly interface to upload an image and see the prediction.

🛠️ Technologies Used
Programming Language: Python 🐍
Deep Learning Framework: TensorFlow/Keras 🤖
Image Processing: OpenCV, Pillow (PIL) 🖼️
Dataset Handling: ImageDataGenerator, JSON 📂
GUI: Tkinter 🖥️
📊 Dataset Details
The dataset consists of Pokémon images organized into folders by class (Pokémon name). It is split into:

Training Set: Used to train the CNN model.
Test Set: Used to evaluate the model’s performance.
Each class represents a unique Pokémon species, and the dataset is automatically labeled using ImageDataGenerator.

🚀 How It Works
1️⃣ Train the CNN Model

Uses Convolutional, Pooling, Batch Normalization, and Dropout layers for feature extraction.
Compiles the model with Adam optimizer and categorical cross-entropy loss.
Saves the trained model as pokemon_cnn_final.h5.
2️⃣ Test the Model

Loads the trained model (pokemon_cnn_final.h5).
Predicts Pokémon from test images and prints confidence scores.
3️⃣ Run the GUI (Pokédex)

Allows users to upload an image and classify Pokémon.
Displays the predicted Pokémon name and confidence score.
📈 Model Performance
The model was trained for 50 epochs, with data augmentation applied to improve accuracy.
Training and validation accuracy is plotted to evaluate performance.

🔮 Future Improvements
Improve Accuracy – Train on a larger dataset and fine-tune hyperparameters.
Optimize Model – Use Transfer Learning (ResNet, MobileNet, etc.) for better performance.
Deploy as a Web App – Convert the GUI into a Flask/Django web application.
💡 Conclusion
This Pokémon Image Classifier demonstrates the power of deep learning in image recognition. By training a CNN model on a Pokémon dataset, we created a system capable of accurately classifying Pokémon from images. 🚀🔍

