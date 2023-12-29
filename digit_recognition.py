import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import google.generativeai as genai
from pathlib import Path

# Configure the generative AI
genai.configure(api_key="your api key")


def process_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()

    if file_path:
        try:
            # Load the selected image using PIL
            image = Image.open(file_path)

            # Resize the image for display
            image.thumbnail((300, 300))

            # Display the original image on the GUI
            original_img = ImageTk.PhotoImage(image)
            original_image_label.configure(image=original_img)
            original_image_label.image = original_img

            # Convert the image to bytes for AI model input
            image_bytes = Path(file_path).read_bytes()

            # Prepare prompt parts for AI model
            image_data = {"mime_type": "image/jpeg", "data": image_bytes}
            prompt_parts = [
                "Recognize the digit in the image:",
                image_data,
            ]

            # Generate content using the image
            response = model.generate_content(prompt_parts)

            # Display the result in a label
            generated_label.config(text=f"The predicted digit is: {response.text}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    else:
        messagebox.showwarning("No File Selected", "Please select an image file.")


def create_gui():
    root = tk.Tk()
    root.title("Digit Recognition")

    global model
    # Set up the generative AI model
    generation_config = {
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }
    model = genai.GenerativeModel(model_name="gemini-pro-vision", generation_config=generation_config)

    # Create a button to choose an image
    select_button = tk.Button(root, text="Select Image", command=process_image)
    select_button.pack()

    # Create a frame to display the original image
    original_image_frame = tk.LabelFrame(root, text="Original Image")
    original_image_frame.pack(side=tk.LEFT, padx=10, pady=10)

    global original_image_label
    original_image_label = tk.Label(original_image_frame)
    original_image_label.pack()

    # Create a frame to display the generated response
    generated_response_frame = tk.LabelFrame(root, text="The digit is:")
    generated_response_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    global generated_label
    generated_label = tk.Label(generated_response_frame, wraplength=300, justify="left")
    generated_label.pack()

    root.mainloop()


if __name__ == "__main__":
    create_gui()