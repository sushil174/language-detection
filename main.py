import pandas as pd
import numpy as np
import sys as sys
import tkinter as tk
from model import Model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

model = Model()
root = tk.Tk()
font = ("Helvetica", 15) 
root.geometry('1000x600')
root.configure(bg="#1f2937")
greeting = tk.Label(text="Language Detection",font=("Helvetica",20,"bold"),foreground="#f9faf8",bg="#1f2937")
root.title("Languge Detection")
label1 = tk.Label(root,text="Enter Text",font=font,foreground="white",bg="#1f2937")
label2 = tk.Label(root,text="Language",font=font,foreground="white",bg="#1f2937")
text = tk.Text(root,width=50,height=5,font=font,foreground="white",bg="#1f2937")
label3 = tk.Label(root,text="language",font=font,foreground="white",bg="#1f2937")


def printOutput():
    user = text.get("1.0","end-1c")
    label3["text"] = model.detect(user)
button = tk.Button(root,text="Detect",font=font,foreground="white",bg="#1f2937",command=lambda:printOutput())


greeting.place(x=400,y=20)
label1.place(x=90,y=100 )
text.place(x=240,y=100)

label2.place(x=90,y=300)
label3.place(x=240,y=300)

button.place(x=240,y=400)

root.mainloop()


