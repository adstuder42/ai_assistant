from speech_recognition import Recognizer, Microphone
import whisper

recognizer = Recognizer()

# On enregistre le son
with Microphone() as source:
    print("Adjusting ambient noise...")
    recognizer.adjust_for_ambient_noise(source)
    print("Listening...")
    recorded_audio = recognizer.listen(source)
    print("Recording complete")
    
# Reconnaissance de l'audio
try:
    print("Speech recognition...")
    text = recognizer.recognize_google(
            recorded_audio, 
            language="fr-FR"
        )
    print("You said : {}".format(text))
except Exception as ex:
    print(ex)




