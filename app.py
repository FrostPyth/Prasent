import os
import subprocess
import json
import threading
import time
import torch

# Determine device for model loading (GPU if available, else CPU)
device_id = 0 if torch.cuda.is_available() else -1

import whisper
import librosa
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for
from transformers import pipeline
from deepface import DeepFace
from pythainlp.tokenize import word_tokenize

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_fallback")
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'webm', 'm4a'}
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
model = whisper.load_model("large")  # adjust size if needed
try:
    emotion_classifier = pipeline(
        "text-classification",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        return_all_scores=False,
        device=device_id
    )
except Exception as e:
    print("Error loading emotion classifier:", e)
    emotion_classifier = None

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(file_path):
    wav_path = file_path.rsplit('.', 1)[0] + ".wav"
    subprocess.run(f'ffmpeg -i "{file_path}" -ar 16000 -ac 1 "{wav_path}"',
                   shell=True, check=True)
    return wav_path

def transcribe_speech(audio_path):
    try:
        result = model.transcribe(
            audio_path,
            language="th",
            beam_size=5,
            best_of=5,
            temperature=0.0
        )
        return result.get("text", "")
    except Exception as e:
        print("Transcription Error:", e)
        return "Error during transcription."

def analyze_text_emotion(text):
    if emotion_classifier is None:
        return "ไม่สามารถวิเคราะห์อารมณ์ได้"
    try:
        result = emotion_classifier(text)
        label = result[0]['label']
        stars = int(label[0])
        if stars <= 2:
            return "ลบ"
        elif stars == 3:
            return "เป็นกลาง"
        else:
            return "บวก"
    except Exception as e:
        print("Text Emotion Analysis Error:", e)
        return "Error during emotion analysis."

def detect_filler_words(text):
    words = word_tokenize(text, engine='newmm')
    filler_words = ["อืม", "เอ่อ", "อ๊ะ", "อะ", "ครับผม", "ครับ", "ค่ะ", "นะครับ", "นะคะ", "ทำการ", "มีการ", "โดย", "แล้วก็"]
    return sum(1 for w in words if w in filler_words)

def extract_frame(video_path):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        if success:
            frame_path = video_path.rsplit('.', 1)[0] + "_frame.jpg"
            cv2.imwrite(frame_path, frame)
            return frame_path
    except Exception as e:
        print("Error extracting frame:", e)
    return None

def analyze_facial_expression(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        dom = result.get('dominant_emotion', "")
        trans = {
            "happy": "มีความสุข",
            "sad": "เศร้า",
            "angry": "โกรธ",
            "neutral": "นิ่งเฉย",
            "surprise": "ตกใจ",
            "fear": "กลัว",
            "disgust": "ขยะแขยง"
        }
        return trans.get(dom.lower(), dom)
    except Exception as e:
        print("Facial Expression Analysis Error:", e)
        return "ไม่สามารถวิเคราะห์ใบหน้าได้"

def analyze_speech_features(audio_path, transcription):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        words = word_tokenize(transcription, engine='newmm')
        words_count = len(words)
        pace = round((words_count / duration) * 60, 2) if duration > 0 else 0
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        mean_pitch = None
        if f0 is not None:
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                mean_pitch = np.mean(f0_clean)
        if mean_pitch is None:
            tone = "ไม่สามารถวิเคราะห์"
        elif mean_pitch < 150:
            tone = "ต่ำ"
        elif mean_pitch <= 250:
            tone = "กลาง"
        else:
            tone = "สูง"
        return {"pace": pace, "tone": tone}
    except Exception as e:
        print("Error analyzing speech features:", e)
        return {"pace": 0, "tone": "Error"}

def analyze_speaker_confidence(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        rms = librosa.feature.rms(y=y)
        mean_rms = np.mean(rms)
        threshold = 0.08
        confidence = min(100, (mean_rms / threshold) * 100)
        return round(confidence, 2)
    except Exception as e:
        print("Error analyzing speaker confidence:", e)
        return 0

def generate_feedback(results):
    feedback = []
    if results.get('filler_words_count', 0) > 5:
        feedback.append("ลดการใช้คำฟุ่มเฟือยเช่น 'อืม', 'เอ่อ' ให้เหลือน้อยลง")
    try:
        pace = float(results.get('pace', 0))
        if pace < 100:
            feedback.append("คุณพูดช้าเกินไป ลองเพิ่มความเร็วในการพูด")
        elif pace > 150:
            feedback.append("คุณพูดเร็วเกินไป ลองลดความเร็วในการพูด")
        else:
            feedback.append("ความเร็วในการพูดเหมาะสมแล้ว")
    except:
        feedback.append("ไม่สามารถวิเคราะห์ความเร็วในการพูดได้")
    if results.get('emotion') == "ลบ":
        feedback.append("ลองปรับเปลี่ยนอารมณ์ให้ดูเป็นบวกมากขึ้น")
    else:
        feedback.append("อารมณ์ในการพูดดูดีแล้ว")
    if results.get('speaker_confidence', 0) < 50:
        feedback.append("เพิ่มความมั่นใจในการพูดโดยการฝึกซ้อมเพิ่มเติม")
    else:
        feedback.append("ความมั่นใจของคุณดีแล้ว")
    if results.get('facial_expression') not in ["มีความสุข", "นิ่งเฉย", "ไม่สามารถวิเคราะห์ใบหน้าได้"]:
        feedback.append("ปรับปรุงการแสดงออกทางใบหน้าให้เป็นธรรมชาติมากขึ้น")
    else:
        feedback.append("การแสดงออกทางใบหน้าดูดีแล้ว")
    return feedback

def process_transcription(file_path):
    audio_path = file_path if file_path.endswith('.wav') else convert_to_wav(file_path)
    transcription = transcribe_speech(audio_path)
    text_emotion = analyze_text_emotion(transcription)
    speaker_confidence = analyze_speaker_confidence(audio_path)
    speech_features = analyze_speech_features(audio_path, transcription)
    pace = speech_features.get("pace", 0)
    tone = speech_features.get("tone", "Error")
    filler_count = detect_filler_words(transcription)

    ext = file_path.rsplit('.', 1)[1].lower()
    if ext in ["mp4", "webm"]:
        frame_path = extract_frame(file_path)
        facial_expression = analyze_facial_expression(frame_path) if frame_path else "ไม่สามารถสกัดใบหน้าได้"
    else:
        facial_expression = "N/A"

    results = {
        "file_name": os.path.basename(file_path),
        "transcription": transcription,
        "emotion": text_emotion,
        "speaker_confidence": float(speaker_confidence),
        "pace": pace,
        "tone": tone,
        "filler_words_count": filler_count,
        "facial_expression": facial_expression,
    }

    results["feedback"] = generate_feedback(results)

    try:
        p = float(pace)
        pace_score = max(0.0, min(p, 150.0)) / 150.0 * 100.0
    except:
        pace_score = 0.0

    f = filler_count
    filler_score = max(0.0, min(1.0, (10.0 - f) / 10.0)) * 100.0

    try:
        confidence_score = min(100.0, max(0.0, float(speaker_confidence)))
    except:
        confidence_score = 0.0

    emo = text_emotion
    emotion_score = {"บวก": 100.0, "เป็นกลาง": 50.0}.get(emo, 0.0)

    overall = (pace_score + filler_score + confidence_score + emotion_score) / 4.0
    results["overall_score"] = round(overall, 2)

    try:
        temp_file = "results_temp.json"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        os.replace(temp_file, "results.json")
        print("✅ Results successfully saved!")
    except Exception as e:
        print("❌ Error saving results.json:", e)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/courses')
def courses():
    return render_template('courses.html')

@app.route('/course/<int:course_id>')
def course_detail(course_id):
    if course_id in (1,2,3):
        return render_template(f'course{course_id}.html')

@app.route('/advice')
def advice():
    return render_template('advice.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400
    f = request.files['file']
    if f.filename == '':
        return "No file selected", 400
    if f and allowed_file(f.filename):
        dest = os.path.join(
            app.config['UPLOAD_FOLDER'],
            f"{int(time.time())}_{f.filename}"
        )
        f.save(dest)

        if os.path.exists("results.json"):
            os.remove("results.json")
        threading.Thread(target=process_transcription, args=(dest,)).start()

        return redirect(url_for('results_page'))
    return "Invalid file type", 400

@app.route('/check_results')
def check_results():
    ready = os.path.exists("results.json") and os.stat("results.json").st_size > 0
    return jsonify({"ready": ready})

@app.route('/get_results')
def get_results():
    if os.path.exists("results.json"):
        with open("results.json", "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Results not ready"}), 202

@app.route('/results')
def results_page():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
