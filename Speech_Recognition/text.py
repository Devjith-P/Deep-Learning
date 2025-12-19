#pyttsx3
import pyttsx3
#class ==>init()
txt_sp=pyttsx3.init()
#text=input("Enter the text: ")
#txt_sp.say(text)
#txt_sp.runAndWait()

#Female
voice=txt_sp.getProperty('voices')
txt_sp.setProperty('voice',voice[1].id)
txt_sp.setProperty('volume',0.3)
txt_sp.setProperty('rate',160)

txt_sp.say("So. Tell us a bit about Yourself.")
txt_sp.runAndWait()
