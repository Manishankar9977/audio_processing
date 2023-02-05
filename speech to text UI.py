from tkinter import*
from turtle import left
from PIL import ImageTk,Image
import speech_recognition as sr
import IPython.display as ipd
import pyaudio
import wave
root = Tk()
r = sr.Recognizer()
filename = 'r7.wav'
def record():
    with sr.Microphone() as source:
        audio_data = r.listen(source)
        global aud 
        aud = audio_data
        mylabel1=Label(root,text="Recorded..")
        mylabel1.pack()
def play():
    #ipd.Audio(aud)
# Set chunk size of 1024 samples per data frame
    chunk = 1024  

# Open the sound file 
    wf = wave.open(filename)

# Create an interface to PortAudio
    p = pyaudio.PyAudio()

# Open a .Stream object to write the WAV file to
# 'output = True' indicates that the sound will be played rather than recorded
    stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)

# Read data in chunks
    data = wf.readframes(chunk)

# Play the sound by writing the audio data to the stream
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)

# Close and terminate the stream
    stream.close()
    p.terminate()
def convert():
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
        tex= r.recognize_google(audio_data)
        new_window=Toplevel(root)
        new_window.geometry("250x250")
        new_window.title("Converted text")
        new_window.resizable(False,False)
        lbl=Label(new_window,text=tex)
        lbl.pack()


#root.geometry("600x700+400+80")
root.resizable(False,False)
root.title("Audio Recorder and converter")

root.iconbitmap("D:\shravanne-tasks\logo.ico")
#LOGO
my_img=ImageTk.PhotoImage(Image.open("D:\\shravanne-tasks\\logo.jpg"))
lab=Label(image=my_img,background="#CDCDAA")
lab.pack()
#record

 

button1=Button(root,text="Record",bg="#9898F5",fg="black",width="20",command=record)
button2=Button(root,text="Play",bg="#9898F5",fg="black",width="20",command=play)
button3=Button(root,text="Convert to text",bg="#9898F5",fg="black",width="20",command=convert)

button1.pack(pady=40)
button2.pack(pady=40)
button3.pack(pady=40)

root.mainloop()