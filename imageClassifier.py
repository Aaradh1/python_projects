from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input

window = Tk()
window.geometry("600x600")
window.title("ImageClassifier")


def get_img_dir():
    global img_ini
    img_ini = filedialog.askopenfile()
    button_1 = Button(window, text="Classify", width=10, font=("Century Gothic", 10), command=pred_img)
    button_1.place(x=250, y=450)
    retract_img()
    

def retract_img():
    try:
        global img_dir
        img_dir = img_ini.name
        img1 = Image.open(img_dir)
        img_resized = img1.resize((300, 200))
        global img_to_disp
        img_to_disp = ImageTk.PhotoImage(img_resized)
        disp_img()
    except Exception as e:
        label = Label(window, text=f"Please enter an image and try again!\n{e}", fg="red", font=("Century Gothic", 12, "bold")).place(x=150, y=220)


def disp_img():
    label = Label(image=img_to_disp, width=300, height=200) 
    label.place(x=150, y=240)
    
    
def pred_img(): 
    img_to_pred = image.load_img(img_dir, target_size=(224,224))
    cnn = MobileNetV2(include_top=True, weights='imagenet')
    img_arr = image.img_to_array(img_to_pred)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_processed = preprocess_input(img_arr)
    global pred
    pred = cnn.predict(img_processed)
    disp_pred()  
    
    
def disp_pred():
     pred_dec = decode_predictions(pred)
     null, pred_output, null = pred_dec[0][0] 
     to_disp = f"This object(s) probably is a(n)\n{pred_output}"
     label = Label(window, text=to_disp, fg="red", width=25, font=("Century Gothic", 11))
     label.place(x=180, y=500)
     
    
label_0 = Label(window, text="Image Classifier", width=20, relief="solid", font=("Century Gothic", 16, "bold"))
label_0.place(x=170, y=10)

label_0_1 = Label(window, text="Using MobileNetV2 along with Imagenet", width=33, font=("Century Gothic", 11))
label_0_1.place(x=150, y=41)

label_0_1 = Label(window, text="Programmed by Aaradh Nepal", width=33, font=("Century Gothic", 8, "italic"))
label_0_1.place(x=180, y=65)

label_1 = Label(window, text="Upload the image file to be classified", width=35, font=("Century Gothic", 10, "bold"))
label_1.place(x=160, y=130)

button_1 = Button(window, text="Browse", width=10, font=("Century Gothic", 10), command=get_img_dir)
button_1.place(x=250, y=170)

window.mainloop()

