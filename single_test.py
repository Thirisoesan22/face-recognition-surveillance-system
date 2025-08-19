import tkinter as tk
from PIL import Image, ImageTk
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import cv2 as cv
import pickle
from keras.models import load_model, Model
from sklearn.preprocessing import Normalizer, LabelEncoder
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

model= load_model("file/My_Model.h5")
data = np.load("file/ce_dataset.npz")
#dense_model = load_model("E:/FinalYear/SHO/MyProject/Dense_classifier.h5")
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('dropout').output)
dense_model=load_model("file/classifier.h5")

# Initialize global variables
path = None
photo = None
results = None
face=None
img = None
p = None
sp = None
samples = None
FE = None
reshaped_fe = None 
x = None # Initialize img as None

def get_img_path():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg")])
    return file_path

def open_image():
    global path, results, img, face, p , sp, samples, FE, reshaped_fe, x
    image_path = get_img_path()
    if image_path:
        path = image_path
        # Display the selected image on canvas1
        display_image(path)
        # Detect faces in the image
        detector = MTCNN()
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = detector.detect_faces(img)
        face=cropped_img()
        p=img_pixel(face)
        sp=preprocess1(p)
        samples=preprocess2(sp)
        FE=preprocess3(samples)   
        reshaped_fe=reshape_fe(FE)
        x=preprocess4(reshaped_fe)
        return img, results, face, p, FE, reshaped_fe, x  # Return img and results

def display_image(image_path):
    global photo
    image = Image.open(image_path)
    image_width, image_height = image.size
    # Calculate the scale factors
    scale_x = 1.0
    scale_y = 1.0
    # If the image is larger than the canvas, calculate scale factors to fit it
    if image_width > canvas1.winfo_reqwidth():
        scale_x = canvas1.winfo_reqwidth() / image_width
    if image_height > canvas1.winfo_reqheight():
        scale_y = canvas1.winfo_reqheight() / image_height
    scale_factor = min(scale_x, scale_y)
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(resized_image)
    canvas1.delete("all")
    canvas1.create_image(0, 0, anchor=tk.NW, image=photo)


            
def cropped_img():
    global results,img  # Access the global img variable
    for i in results:
        (x, y, w, h) = i['box']
        cropped_image = img[y:y+h, x:x+w]
        pil_image = Image.fromarray(cropped_image)
        pil_image_resized = pil_image.resize((160, 160), Image.ANTIALIAS)
    return pil_image_resized
            
def show_cropped_img(canvas):
    global results
    crop = cropped_img()
    height, width = crop.size
    tk_image = ImageTk.PhotoImage(crop)
    canvas.create_image(35, 20, anchor=tk.NW, image=tk_image)
    canvas.photo = tk_image  # Keep a reference to prevent garbage collection
    label_text = "Cropped Frame ({}x{})".format(width, height)
    canvas.create_text(110, crop.height + 40, text=label_text, fill="black",font=("Helvetica", 12))
    #canvas.create_text(10, 10, anchor=tk.NW, text="Cropped Frame ({}x{})".format(width, height), fill="white")

def img_pixel(faces):
    # Convert the list of pixels to a NumPy array
    pixels_array = np.array(faces)
    return pixels_array


def show_img_pixel(canvas):
    p_array=img_pixel(face)
    label = tk.Label(canvas, text=str(p_array[:1,:15])+'\n\n\n'+'Shape:'+str(p_array.shape), font=("Helvetica", 14))
    label.pack()
    
def preprocess1(pixels):
    std_pixels=[]
    fpixels = pixels.astype('float32')
    mean, std = fpixels.mean(), fpixels.std()
    std_pixels = ((fpixels - mean) / std)
    return std_pixels

def show_preprocess1(canvas):
    std_pixels=preprocess1(p)
    fp = np.clip(std_pixels, 0, 1)
    #nl=((std_pixels+ 2) / 4) * 255
    nl = (fp*255).astype('uint8')
    image_pil = Image.fromarray(nl)
    image_tk = ImageTk.PhotoImage(image_pil)
    canvas.create_image(10, 20, anchor=tk.NW, image=image_tk)
    canvas.image = image_tk
    label_text = '\n\n\n\n Shape:'+str(std_pixels.shape)
    canvas.create_text(83, image_pil.height + 10, text=label_text, fill="black",font=("Helvetica", 14))
    #canvas.create_text(10, 10, anchor=tk.S, text= "\n Shape:"+str(std_pixels.shape), fill="white")
    #label = tk.Label(canvas, text='\n\n\n\n Shape:'+str(std_pixels.shape), font=("Helvetica", 9))
    #label.pack(pady=50)

def preprocess2(pp):
    samples = np.expand_dims(pp, axis=0)
    return samples

def show_preprocess2(canv):
    sam=preprocess2(sp)
    fp = np.clip(sam, 0, 1)
    sam2 = (fp.squeeze()*255).astype(np.uint8) 
    image_pil = Image.fromarray(np.uint8(sam2))
    image_tk = ImageTk.PhotoImage(image_pil)
    canv.create_image(40, 20, anchor=tk.NW, image=image_tk)
    canv.image = image_tk
    label_text = '\n\n\n\n Shape:'+str(sam.shape)
    canv.create_text(120, image_pil.height + 10, text=label_text, fill="black",font=("Helvetica", 14))
    
    #first_few_values = sam.flatten()[:45]
    #reshaped_values = first_few_values.reshape(15, 3)
    #values_text = '\n ['.join(' '.join(f'{value:.6f}' for value in row)+']' for row in reshaped_values)
    #label_text = f" {values_text}"
    #label = tk.Label(canv, text='[  '+'[[['+label_text+']]'+'   ]\n\n'+'Shape:'+ str(sam.shape), font=("Helvetica", 9))
    #label.pack()
    
def preprocess3(ss):
    yhat = feature_extractor.predict(ss)
    face_embedded = yhat[0]
    return face_embedded

def show_preprocess3(canv7):
    p3=preprocess3(samples)
    first_values = p3[:45]
    reshape_first_values = first_values.reshape(15, 3)

    text = '\n'.join(' '.join(f'{value:.5f}' for value in row) for row in reshape_first_values)
    label = tk.Label(canv7, text=text + '\n\n' + 'Shape:'+ str(p3.shape), font=("Helvetica", 10))
    label.pack()

def reshape_fe(fe):
    FE2=fe.reshape(1, -1)
    return FE2

def show_reshape_fe(canv8):
    RFE=reshape_fe(FE)
    first_few_values = RFE.flatten()[:45]
    
    # Reshape the first few values into a 15x3 array
    reshape_first_values = first_few_values.reshape(15, 3)
    
    # Format the reshaped array into a text representation
    text = '\n'.join(' '.join(f'{value:.5f}' for value in row) for row in reshape_first_values)
    
    # Display the formatted text along with the shape of X
    label = tk.Label(canv8, text=text+'\n\n'+'Shape:'+str(RFE.shape), font=("Helvetica", 10))
    label.pack()

def preprocess4(R_fe):
    in_encoder = Normalizer(norm='l2')
    X = in_encoder.transform(R_fe)
    return X

def show_preprocess4(canv99):
    XX=preprocess4(reshaped_fe)
    # Take the first 45 elements for reshaping (assuming X has more than 45 elements)
    first_few_values = XX.flatten()[:45]
    
    # Reshape the first few values into a 15x3 array
    reshape_first_values = first_few_values.reshape(15, 3)
    
    # Format the reshaped array into a text representation
    text = '\n'.join(' '.join(f'{value:.5f}' for value in row) for row in reshape_first_values)
    
    # Display the formatted text along with the shape of X
    label = tk.Label(canv99, text=text+'\n\n'+'Shape:'+str(XX.shape), font=("Helvetica", 10))
    label.pack()

def face_cnn_classifier(A):
    """Preprocessed images, classifies and
    returns predicted label and probability"""
    yhat = dense_model.predict(A)
    predicted_class_index = np.argmax(yhat)  # Get the index of the class with the highest probability
    probability = yhat[0][predicted_class_index]  # Get the probability of the predicted class

    trainy = data['arr_1']
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    predicted_class_label = out_encoder.inverse_transform([predicted_class_index])  # Decode the predicted class

    label = predicted_class_label[0]
    if probability < 0.7:
        return "unknown", str(probability)
    return label, str(probability)

def classify2(c,canv):
    label, probability = face_cnn_classifier(c)
    show_cropped_img(canv)
    label_text = f"{label}\nProbability: {probability}"
    text_x, text_y = 10, 185  # Position for the label text
    text_width, text_height = 210, 60  # Size of the text box
    
    # Draw background highlight (rectangle)
    canv.create_rectangle(text_x, text_y, text_x + text_width, text_y + text_height, fill="yellow")
    canv.create_text(text_x + 10, 195, anchor=tk.NW, text=label_text, fill="Purple", font=("Helvetica", 14))
    

def classify_result2(canvas10):    
    classify2(x,canvas10)
    
    
#end of functions   

            
def btn_3():
    global img, results
    if img is not None and results:
        show_cropped_img(canvas3)
        
def btn_4():
    show_preprocess1(canvas4)
    
def btn_5():
    show_preprocess2(canvas5)
    
def btn_6():
    show_preprocess3(canvas6)
    
def btn_7():
    show_reshape_fe(canvas7)
    
def btn_8():
    show_preprocess4(canvas8)
    

def btn_10():
    classify_result2(canvas10)

def reset_application():
    # Clear canvas contents
    canvas1.delete("all")
    canvas2.delete("all")
    canvas3.delete("all")
    canvas4.delete("all")
    canvas5.delete("all")
    canvas6.delete("all")
    canvas7.delete("all")
    canvas8.delete("all")
    canvas10.delete("all") 
    # Reset labels (assuming labels are stored in a list or dictionary)
    clear_labels(canvas2)  # Custom function to clear labels
    clear_labels(canvas6)
    clear_labels(canvas7)
    clear_labels(canvas8)

def clear_labels(container_widget):
    # Clear labels within a container widget
    for widget in container_widget.winfo_children():
        if isinstance(widget, tk.Label):
            widget.destroy()
            
def draw_bounding_box(image, results):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for result in results:
        x, y, width, height = result['box']
        rect = plt.Rectangle((x, y), width, height, fill=False, color='cyan', linewidth=2.6)
        ax.add_patch(rect)
    ax.axis('off')
    return fig

def show_bounding_box(canvas):
    global img, results
    if img is not None and results:
        fig = draw_bounding_box(img, results)
        
        # Convert the Matplotlib figure to a PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        pil_image = Image.fromarray(img_array)
        
        # Resize the image to fit the canvas
        canvas.update()  # Ensure the canvas dimensions are updated
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        image_width, image_height = pil_image.size
        
        # Calculate scaling factors
        scale_x = canvas_width / image_width
        scale_y = canvas_height / image_height
        scale_factor = min(scale_x, scale_y)
        
        # Adjust the scale factor to increase the size (e.g., up to 10%)
        scale_factor *= 2.1
        
        # New dimensions
        new_width = int(image_width * scale_factor)
        new_height = int(image_height * scale_factor)
        
        # Make sure the resized image fits within the canvas
        if new_width > canvas_width:
            new_width = canvas_width
            new_height = int(image_height * (canvas_width / image_width))
        if new_height > canvas_height:
            new_height = canvas_height
            new_width = int(image_width * (canvas_height / image_height))
        
        # Resize image
        resized_image = pil_image.resize((new_width*2, new_height*2), Image.ANTIALIAS)
        tk_image = ImageTk.PhotoImage(resized_image)
        
        # Display the image with the bounding box on the canvas
        canvas.delete("all")  # Clear previous content
        canvas.create_image(110, 130, anchor=tk.CENTER, image=tk_image)
        canvas.image = tk_image  # Keep a reference to prevent garbage collection

        
def handle_key_event(event):
    # Check if the key combination Ctrl+t is pressed
    if event.state == 4 and event.keysym == 't':
        # Ctrl is pressed (event.state == 4) and 't' key is pressed (event.keysym == 't')
        reset_application()

# end of functions        
# Create the main window
window = tk.Tk()
window.title("Face Classification")
window.geometry("1250x550")

canvas1 = tk.Canvas(window, width=220, height=260)
canvas1.grid(row=0, column=0, padx=10, pady=10)
button1 = tk.Button(window, text="Upload a Photo",font=("Helvetica", 15),bg="cyan",fg="black", command=open_image)
button1.grid(row=1, column=0,sticky="n")

canvas2 = tk.Canvas(window, width=220, height=260)
canvas2.grid(row=0, column=1, padx=0, pady=0)
button2 = tk.Button(window, text="Detect Face",font=("Helvetica", 15),bg="cyan",fg="black", command=lambda: show_bounding_box(canvas2))
button2.grid(row=1, column=1)

canvas3 = tk.Canvas(window, width=220, height=260)
canvas3.grid(row=0, column=2, padx=60, pady=10)
button3 = tk.Button(window, text="Crop and Resize",font=("Helvetica", 15),bg="cyan",fg="black",command=btn_3)
button3.grid(row=1, column=2, sticky="n")

canvas4 = tk.Canvas(window, width=170, height=260)
canvas4.grid(row=0, column=3, padx=20, pady=10)
button4 = tk.Button(window, text="Standardize Image",font=("Helvetica", 15),bg="cyan",fg="black", command=btn_4)
button4.grid(row=1, column=3)

canvas5 = tk.Canvas(window, width=250, height=260)
canvas5.grid(row=0, column=4, padx=10, pady=10)
button5 = tk.Button(window, text="Reshape",font=("Helvetica", 15),bg="cyan",fg="black", command=btn_5)
button5.grid(row=1, column=4)

canvas6 = tk.Canvas(window, width=220, height=270)
canvas6.grid(row=2, column=0, padx=10, pady=20, sticky="s")
button6 = tk.Button(window, text="Extract Feature",font=("Helvetica", 15),bg="cyan",fg="black", command=btn_6)
button6.grid(row=3, column=0,sticky="n")

canvas7 = tk.Canvas(window, width=240, height=270)
canvas7.grid(row=2, column=1, padx=0, pady=20, sticky="s")
button7 = tk.Button(window, text="Reshape",font=("Helvetica", 15),bg="cyan",fg="black",command=btn_7)
button7.grid(row=3, column=1,sticky="n")

canvas8 = tk.Canvas(window, width=220, height=270)
canvas8.grid(row=2, column=2, padx=0, pady=20, sticky="s")
button8 = tk.Button(window, text="Normalization",font=("Helvetica", 15),bg="cyan",fg="black",command=btn_8)
button8.grid(row=3, column=2,sticky="n")

canvas10 = tk.Canvas(window, width=220, height=270)
canvas10.grid(row=2, column=3, columnspan=2, padx=30, pady=30, sticky="n")
button10 = tk.Button(window, text="Classify",font=("Helvetica", 15),bg="cyan",fg="black", command=btn_10)
button10.grid(row=3, column=3,columnspan=2, sticky="s")

# Bind key event to handle Ctrl+t (optional)
window.bind('<Control-t>', handle_key_event)

# Start the main event loop
window.mainloop()

