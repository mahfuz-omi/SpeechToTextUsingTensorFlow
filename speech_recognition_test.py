# pip install SpeechRecognition
# import e all small letter e likhte hoy
#pip install e shob capital letter
import speech_recognition as sr
# https://medium.com/@rahulvaish/speech-to-text-python-77b510f06de

#recognize_bing( )
#recognize_google( )
#recognize_google_cloud( )
#recognize_houndify( )
#recognize_ibm( )
#recognize_wit( )
#recognize_sphinx( )


recognizer = sr.Recognizer()

print('say something....')
with sr.Microphone() as source:
    try:
        audio = recognizer.listen(source)
    except:
        print('exception')

try:
    # by default,language is en-us
    print('text: ',recognizer.recognize_google(audio,language='bn'))
    # text:  কেমন আছো তুমি ভালো আছো
    # no punctuation
except:
    print("i can't understand")

