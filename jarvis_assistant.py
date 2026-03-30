import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import requests
import pyttsx3
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import tempfile
import wave
import time

# ==============================
# SETTINGS
# ==============================
MODEL_NAME = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"
WHISPER_MODEL_SIZE = "medium"   # tiny / base / small / medium

# ==============================
# INIT MODELS
# ==============================
print("Loading Whisper model...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE)

print("Initializing Speaker...")
engine = pyttsx3.init()
engine.setProperty("rate", 195)

conversation_memory = []

is_listening = False
speech_queue = queue.Queue()

# ==============================
# SPEAK FIX (NO LOOP ERROR)
# ==============================
def speak(text):
    def run():
        try:
            engine.stop()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("Speaker Error:", e)

    threading.Thread(target=run, daemon=True).start()

# ==============================
# RECORD AUDIO
# ==============================
def record_audio(duration=4, fs=16000):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(fs)
        f.writeframes((audio * 32767).astype(np.int16).tobytes())

    return temp_file.name

# ==============================
# TRANSCRIBE WITH WHISPER
# ==============================
def transcribe(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ""
    for segment in segments:
        text += segment.text
    return text.strip().lower()

# ==============================
# STREAMING LLM
# ==============================
def ask_llm_stream(prompt):
    global conversation_memory

    conversation_memory.append({"role": "user", "content": prompt})

    full_prompt = ""
    for msg in conversation_memory[-6:]:
        full_prompt += f"{msg['role']}: {msg['content']}\n"

    response_text = ""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": True
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                if '"response":"' in data:
                    part = data.split('"response":"')[1].split('"')[0]
                    response_text += part
                    chat_area.insert(tk.END, part)
                    chat_area.see(tk.END)
                    root.update()

        conversation_memory.append({"role": "assistant", "content": response_text})

    except Exception as e:
        response_text = f"\nLLM Error: {e}"

    return response_text

# ==============================
# LISTEN LOOP
# ==============================
def listen_loop():
    global is_listening

    while is_listening:
        status_label.config(text="🎤 Listening...", fg="cyan")

        audio_path = record_audio()
        text = transcribe(audio_path)

        if text:
            chat_area.insert(tk.END, f"\n You: {text}\n🤖 AI: ")
            chat_area.see(tk.END)

            answer = ask_llm_stream(text)
            speak(answer)

        time.sleep(0.3)

# ==============================
# START / STOP
# ==============================
def start_listening():
    global is_listening
    if not is_listening:
        is_listening = True
        threading.Thread(target=listen_loop, daemon=True).start()

def stop_listening():
    global is_listening
    is_listening = False
    status_label.config(text=" Stopped", fg="red")

# ==============================
# UI
# ==============================
root = tk.Tk()
root.title("JARVIS ULTRA AI")
root.geometry("950x650")
root.configure(bg="#0f172a")

title = tk.Label(root, text="JARVIS ULTRA AI",
                 font=("Arial", 26, "bold"),
                 fg="cyan", bg="#0f172a")
title.pack(pady=15)

status_label = tk.Label(root, text="Idle",
                        font=("Arial", 12),
                        fg="white", bg="#0f172a")
status_label.pack()

button_frame = tk.Frame(root, bg="#0f172a")
button_frame.pack(pady=10)

start_btn = tk.Button(button_frame, text="▶ Start",
                      bg="green", fg="white",
                      width=15, command=start_listening)
start_btn.pack(side=tk.LEFT, padx=10)

stop_btn = tk.Button(button_frame, text="■ Stop",
                     bg="red", fg="white",
                     width=15, command=stop_listening)
stop_btn.pack(side=tk.LEFT, padx=10)

chat_area = scrolledtext.ScrolledText(
    root,
    bg="#1e293b",
    fg="white",
    font=("Consolas", 12)
)
chat_area.pack(fill="both", expand=True, padx=20, pady=20)

root.mainloop()