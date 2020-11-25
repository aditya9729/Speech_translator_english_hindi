from flask import Flask, render_template, request, redirect,json
import speech_recognition as sr
from gtts import gTTS
from src.translate import Translator
import os

MODEL_PATH='data'

app = Flask(__name__)

translator = Translator(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    """Deploys model using flask api - uses ASR, NMT and TTS all in one and renders template"""
    transcript = ""
    translated=""
    audio1,audio2="",""
    if request.method == "POST":
        file = request.files["file"]
        if "file" not in request.files:
            return redirect(request.url)


        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                recognizer.adjust_for_ambient_noise(source)
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)
            translation = translator.translate(source='en', target='hi', text=transcript)
            translated = translation[0]
            print(translated)
            obj = gTTS(text=translation[0], slow=False, lang='hi')
            audio2 = "output.mp3"
            obj.save(os.path.join('static','output.mp3'))


    return render_template('index.html', transcript=json.dumps({'english':transcript}),translated=json.dumps({'hindi':translated}, ensure_ascii=False).encode('utf8').decode())




if __name__=='__main__':
    app.run(host="0.0.0.0",port=5000, debug=True, threaded=True)

