import pyttsx3
newVoiceRate = 100
text_speech = pyttsx3.init()
text_speech.setProperty('rate', newVoiceRate)

text_speech.say("my name is saikat, I would like to make you happy")

text_speech.runAndWait()

