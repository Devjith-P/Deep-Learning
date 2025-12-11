#pip install pyaudio
#pip install SpeechRecognition

import speech_recognition as sr
import pyttsx3 as pt

engine = pt.init()
voice = engine.getProperty('voices')
engine.setProperty('voice',voice[1].id)
engine.setProperty('rate',150)
engine.setProperty('volume',0.2)

def speech_txt():
    r=sr.Recognizer()     #init
    while True:
        with sr.Microphone() as source:
             #source=sr.Microphone()
            print("Speak Now........")
            audio=r.listen(source)
        
        try:
            text=r.recognize_google(audio)
            print("You said",text)
            engine.say(text)
            engine.runAndWait()
        except:
            print("Didn't hear anything")
        if 0xff == ord('q'):
            break
speech_txt()
    
        