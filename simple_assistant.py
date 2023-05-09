#!/usr/bin/env python3

# pip install -U openai-whisper
# sudo apt update && sudo apt install ffmpeg
# pip install setuptools-rust
# pip install openai
# pip install gTTS
# pip install playsound

from speech_recognition import Recognizer, Microphone
import whisper
import os
import openai
from gtts import gTTS
from playsound import playsound



recognizer = Recognizer()
openai.api_key_path = "./APIkey"
openai.api_key = os.getenv("OPENAI_API_KEY")

# On enregistre le son


def recordSpeech():
    if os.path.isfile("./speech.wav"):
        os.remove("./speech.wav")

    with Microphone() as source:
        print("Adjusting ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        recorded_audio = recognizer.listen(source)
        print("Recording complete")

    with open("speech.wav", "wb") as f:
        f.write(recorded_audio.get_wav_data())

    return


def getTranscript():
    model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("speech.wav")
    audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    print(result.text)
    return (result.text)


def sendPrompt(transcript):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": transcript}]
    )
    # print(completion)
    return (completion.choices[0].message.content)


def playVoice(response):
    language = 'fr'
    myobj = gTTS(text=response, lang=language, slow=False)

    if os.path.isfile("./reponse.mp3"):
         os.remove("./response.mp3")

    myobj.save("response.mp3")
    playsound('./response.mp3')


recordSpeech()
transcript = getTranscript()
response = sendPrompt(transcript)
print("\n\n" + response)
playVoice(response)
