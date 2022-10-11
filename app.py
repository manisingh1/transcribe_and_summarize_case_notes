import json
import os
import speech_recognition as sr
import openai
import whisper as ws
import io
import tempfile
from pydub import AudioSegment
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "temp.wav")



# Record audio
def record_audio_and_transcribe():
    print("starting transcribe")
    with sr.Microphone(sample_rate=16000) as source:
        ai = ws.load_model("tiny.en")
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.pause_threshold = 1
        r.dynamic_energy_threshold = False
        audio = r.listen(source)
        data = io.BytesIO(audio.get_wav_data())
        audio_clip = AudioSegment.from_file(data)
        audio_clip.export(save_path, format="wav")
        result = ai.transcribe(save_path,language='english')
        predicted_text = result["text"]
        print("You said: " + predicted_text)
        return predicted_text
def fake_transcription():
    text = """The Pt. is a 50 y.o. male complaining of substernal chest pain and nausea. The complaint is
described as a heavy pressure mid-sternum with radiation to the left shoulder."""
    return text
    
def make_summary(transcription):
    prompt = "Summarize this for a second-grade student:\n\n" + transcription + "\n\nTl;dr"
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.8,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    # print(response)
    return response.choices[0].text

def handleTranscription():
    audio_transcription = ""
    # mode = "debug"
    mode = "prod"
    if mode == "debug":
        audio_transcription = fake_transcription()
    else:
        audio_transcription = record_audio_and_transcribe()
    return audio_transcription

@app.route("/", methods=("GET", "POST"))
def index():

    if request.method == "POST":
        results = {}
        # record transcription or spoof for testing
        audio_transcription = handleTranscription()
        print(f"Transcribed this: {audio_transcription}")

        # make summary
        summary = make_summary(audio_transcription)
        print(f"Summarized this: {summary}")
        results["transcription"] = audio_transcription
        results["summary"] = summary
        return redirect(url_for("index", results=json.dumps(results)))


    results = request.args.get("results")
    if results:
        res_dict = json.loads(results)
        return render_template("index.html", results=res_dict)

    return render_template("index.html", results=results)
