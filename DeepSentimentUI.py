from Tkinter import *
import tkFileDialog
from PIL import Image, ImageTk
import Tkconstants
import os
import Controller
result = " "

def askopenfile():
  dirname = tkFileDialog.askopenfile()
  if dirname:
    var.set(dirname)

def UserFileInput(status,name):
  optionFrame = Frame(root)
  optionLabel = Label(optionFrame)
  #optionLabel["text"] = name
  optionLabel.pack(side=LEFT)
  text = status
  var = StringVar(root)
  var.set(text)
  w = Entry(optionFrame, textvariable= var)
  w.pack(side = LEFT)
  optionFrame.pack()
  return w, var

def resultWindow(result):
    print "Inside the sub window"
    subWindowFrame = Toplevel()
    subWindowFrame.wm_title("Result")
    subWindow = Label(subWindowFrame, text=result)
    subWindow.pack(side=LEFT, fill="both", expand=True, padx=50, pady=25)

def Print_entry():
  global result
  print var.get().split()[2][1:-2]
  stubpara = var.get().split()[2][1:-2]
  retval = Controller.main(var.get().split()[2][1:-2])
  print('@@@@@@@@@@@@@@@')
  print(retval)
  resultWindow(retval)

if __name__ == '__main__':
  global result
  root = Tk()
  root.title("Speech Emotion Recognition")
  
  #Setting the background Image
  bkImage =Image.open('/home/vyassu/speech-recognition.png')
  bkgdImage = ImageTk.PhotoImage(bkImage)

  # Setting the Initial shape of the UI
  root.geometry('%dx%d+0+0' % (1280,600))

  # Creating Label to store the Background Image
  imageLabel = Label(root,image=bkgdImage)
  imageLabel.place(relx=0.1, rely=0.1,relwidth=1, relheight=1)
  imageLabel.pack(fill=BOTH, expand=YES,side=TOP)

  # options for buttons
  button_opt = {'fill': Tkconstants.BOTH, 'padx': 5, 'pady': 5}


  dirBut = Button(root, text='Choose file', command = askopenfile)
  dirBut.pack()
  getBut = Button(root, text='Start Analysis', command = Print_entry)
  getBut.pack(side = BOTTOM)
  
  # Scroll if necessary
  '''
  tx = Text(master=root)
  tx.pack()

  tx.insert(END, result)
  tx.see(END) 
  '''
  w, var = UserFileInput("", "Filename")

  root.mainloop()
