import os
import numpy as np
from tkinter import Tk, Label, Button, filedialog, messagebox, Canvas
from PIL import Image, ImageTk
from minisom import MiniSom

# -------- Global Variables --------
uploaded_image_vector = None
imgtk = None
som = None
train_images = None
train_labels = None
canvas = None


# -------- Load Images and Train SOM --------
def load_images_from_folder(folder, size=(64, 64)):
    images = []
    labels = []
    for label in ['man', 'dog', 'cat']:
        class_folder = os.path.join(folder, label)
        for filename in os.listdir(class_folder):
            path = os.path.join(class_folder, filename)
            try:
                # Handle palette images with transparency safely
                img = Image.open(path).convert("RGBA").convert('L')
                img = img.resize(size)
                images.append(np.array(img).flatten() / 255.0)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return np.array(images), np.array(labels)

def train_som_model(images):
    som_model = MiniSom(10, 10, images.shape[1], sigma=1.0, learning_rate=0.5)
    som_model.random_weights_init(images)
    som_model.train_random(images, 100)  # 100 iterations
    return som_model


# -------- GUI Functions --------
def browse_image():
    global uploaded_image_vector, imgtk
    file_path = filedialog.askopenfilename(title="Select an image")
    if file_path:
        try:
            img = Image.open(file_path).convert("RGBA").convert("L").resize((64, 64))
            uploaded_image_vector = np.array(img).flatten() / 255.0
            img_display = img.resize((200, 200))
            imgtk = ImageTk.PhotoImage(img_display)
            canvas.create_image(0, 0, anchor='nw', image=imgtk)
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load image: {e}")

def classify_image():
    if uploaded_image_vector is None:
        messagebox.showwarning("Warning", "Please upload an image first.")
        return
    try:
        winner = som.winner(uploaded_image_vector)
        distances = [np.linalg.norm(np.array(som.winner(v)) - np.array(winner)) for v in train_images]
        nearest_index = np.argmin(distances)
        predicted_label = train_labels[nearest_index]
        messagebox.showinfo("Classification Result", f"✅ The image is classified as: {predicted_label.upper()}")
    except Exception as e:
        messagebox.showerror("Classification Error", f"Could not classify the image: {e}")

def start_gui():
    global canvas
    root = Tk()
    root.title("SOM Image Classifier")

    Label(root, text="Upload an image to classify (man, dog, cat):").pack(pady=5)

    canvas = Canvas(root, width=200, height=200)
    canvas.pack()

    Button(root, text="Browse Image", command=browse_image).pack(pady=5)
    Button(root, text="Classify", command=classify_image).pack(pady=5)

    root.mainloop()

    # -------- Main Entry --------
if __name__ == "__main__":
    image_folder = "./images"  # Expecting ./images/man, ./images/dog, ./images/cat
    train_images, train_labels = load_images_from_folder(image_folder)
    if len(train_images) == 0:
        print("No training images found. Check if './images/man', './images/dog', './images/cat' contain images.")
    else:
        som = train_som_model(train_images)
        start_gui()

