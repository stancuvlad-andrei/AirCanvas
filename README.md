# 🎨 AirCanvas AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-AI%20Powered-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/github/license/stancuvlad-andrei/AirCanvas?style=for-the-badge)

> **Draw in the air using a blue marker and let AI guess what you wrote!** ✨

---

## 🖼️ Demo

![AirCanvas Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3Z5a3Z5a3Z5a3Z5a3Z5a3Z5a3Z5a3Z5a3Z5a3Z5a3Z5a3Z5/placeholder.gif)

## 🚀 About The Project

**AirCanvas** is a smart digital drawing board built with **OpenCV** and **TensorFlow**. Unlike standard drawing apps, this project uses a physical blue object (like a pen cap) as your stylus. 

It features an intelligent **Digit & Character Recognizer** trained on the EMNIST dataset, allowing you to write on the screen and have the AI convert your handwriting into text!

### ✨ Key Features

* **🖊️ Color Tracking**: Uses HSV color detection to track a blue marker/object.
* **🤖 AI Recognition**: Includes a CNN model (`DigitRecognizer`) to predict handwritten characters (0-9, A-Z).
* **🎨 Advanced Tools**:
    * **Magic Mode**: Draws in a shifting rainbow pattern.
    * **Spray Can**: Simulates a spray paint effect.
    * **Shape Detector**: Automatically cleans up rough drawings into perfect Triangles, Rectangles, or Circles.
* **🖥️ Dual Display**: Opens two windows—one with the camera UI for the "Teacher" and one clean black canvas for the "Projector/Students".

---

## 🛠️ Tech Stack

* **[Python](https://www.python.org/)**
* **[OpenCV](https://opencv.org/)**: Image processing (HSV, Contours).
* **[TensorFlow/Keras](https://www.tensorflow.org/)**: Convolutional Neural Network (CNN) for character recognition.
* **[NumPy](https://numpy.org/)**: Matrix operations.
* **[EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)**: Dataset used for training the AI model.

---

## 💻 Getting Started

### Prerequisites

You need Python installed along with the required libraries.

```bash
pip install -r requirements.txt