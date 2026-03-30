import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("brain_tumor_model.h5")

# Prediction function
def predict_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    pred = model.predict(img)
    result = np.argmax(pred)
    confidence = np.max(pred) * 100

    if result == 1:
        return f"Tumor Detected ({confidence:.1f}%)", "#ff4d4d"
    else:
        return f"No Tumor ({confidence:.1f}%)", "#4CAF50"

# Upload handler
def upload_image():
    file_path = filedialog.askopenfilename()
    
    if file_path:
        show_result_screen(file_path)

# Switch to result screen
def show_result_screen(file_path):
    for widget in root.winfo_children():
        widget.destroy()

    # Title
    title = tk.Label(root, text="Result",
                     font=("Segoe UI", 18, "bold"),
                     bg="#f5f6fa", fg="#2f3640")
    title.pack(pady=15)

    # Image
    img = Image.open(file_path)
    img = img.resize((280, 280))
    img = ImageTk.PhotoImage(img)

    img_label = tk.Label(root, image=img, bg="#f5f6fa")
    img_label.image = img
    img_label.pack(pady=10)

    # Prediction
    result, color = predict_image(file_path)
    result_label = tk.Label(root, text=result,
                            font=("Segoe UI", 16, "bold"),
                            bg="#f5f6fa", fg=color)
    result_label.pack(pady=15)

    # Back button
    back_btn = tk.Button(root, text="⬅ Back",
                         command=show_home,
                         font=("Segoe UI", 10),
                         bg="#407BFF", fg="white",
                         bd=0, padx=10, pady=5)
    back_btn.pack()

# Home screen
def show_home():
    for widget in root.winfo_children():
        widget.destroy()

    # Title
    title = tk.Label(root, text="Brain Tumor Detection",
                     font=("Segoe UI", 20, "bold"),
                     bg="#f5f6fa", fg="#2f3640")
    title.pack(pady=40)

    # Subtitle
    subtitle = tk.Label(root, text="Upload MRI image to detect tumor",
                        font=("Segoe UI", 11),
                        bg="#f5f6fa", fg="gray")
    subtitle.pack(pady=5)

    # Upload button
    upload_btn = tk.Button(root, text="Upload Image",
                           command=upload_image,
                           font=("Segoe UI", 12, "bold"),
                           bg="#407BFF", fg="white",
                           bd=0, padx=15, pady=8)
    upload_btn.pack(pady=30)

    # Footer
    footer = tk.Label(root, text="AI Medical Assistant",
                      font=("Segoe UI", 9),
                      bg="#f5f6fa", fg="gray")
    footer.pack(side="bottom", pady=10)

# Main window
root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("420x550")
root.configure(bg="#f5f6fa")

# Start with home screen
show_home()

root.mainloop()