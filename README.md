# Image Classification using Self-Organizing Maps (SOM)

This project demonstrates an image classification system developed using **Self-Organizing Maps (SOM)**, an unsupervised neural network architecture. It was built as part of my **Post Graduate Diploma (PGD) in Data Science**.

### 🌟 Features:
- **Unsupervised Learning:** Uses the `minisom` library to cluster and classify images.
- **Multi-class Support:** Capable of identifying images of Humans (Man), Dogs, and Cats.
- **Interactive GUI:** Built with `Tkinter` for easy image uploading and real-time classification.
- **Image Processing:** Implements grayscale conversion, resizing, and normalization using `PIL` and `NumPy`.

### 🛠️ Tech Stack:
- Python
- MiniSom (Self-Organizing Maps)
- NumPy (Data Manipulation)
- Pillow (Image Processing)
- Tkinter (GUI Development)

### 📂 Dataset Structure:
To run this project, organize your images as follows:
./images/
├── man/
├── dog/
└── cat/


### 🚀 How to Run:
1. Install dependencies: `pip install minisom numpy Pillow`
2. Place your training images in the `./images` folder.
3. Run the script: `python SOM.py`
