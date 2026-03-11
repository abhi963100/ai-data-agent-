import speech_recognition as sr
import pyttsx3
import requests
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

# ----------------------------
# INITIALIZE
# ----------------------------
recognizer = sr.Recognizer()
engine = pyttsx3.init()

WAKE_WORD = "hi"

recognizer.pause_threshold = 0.4
recognizer.dynamic_energy_threshold = True
engine.setProperty("rate", 180)

# ----------------------------
# SPEAK FUNCTION
# ----------------------------
def speak(text):
    output_box.insert(tk.END, f"AI: {text}\n")
    output_box.see(tk.END)
    engine.say(text)
    engine.runAndWait()

# ----------------------------
# LLM FUNCTION
# ----------------------------
def ask_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        return response.json()["response"]
    except Exception as e:
        return "Cannot connect to AI model."

# ----------------------------
# LISTEN FUNCTION
# ----------------------------
def listen():
    try:
        with sr.Microphone() as source:
            status_label.config(text="Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
            text = recognizer.recognize_google(audio).lower()

            output_box.insert(tk.END, f"You: {text}\n")
            output_box.see(tk.END)

            return text

    except sr.WaitTimeoutError:
        status_label.config(text="No speech detected")
        return None

    except sr.UnknownValueError:
        status_label.config(text="Could not understand")
        return None

    except Exception as e:
        status_label.config(text="Mic Error")
        return None

# ----------------------------
# MAIN ASSISTANT LOGIC
# ----------------------------
def run_assistant():
    text = listen()

    if not text:
        return

    if WAKE_WORD in text:
        speak("Yes Abhinav, how can I help you?")
        command = listen()

        if not command:
            speak("I did not hear anything.")
            return

        if "exit" in command:
            speak("Goodbye Abhinav.")
            root.quit()
            return

        answer = ask_llm(command)
        speak(answer)
    else:
        speak("Please say hi to activate me.")

# Run in background thread
def start_thread():
    thread = threading.Thread(target=run_assistant)
    thread.start()

# ----------------------------
# GUI WINDOW
# ----------------------------
root = tk.Tk()
root.title("AI Voice Assistant")
root.geometry("500x500")
root.configure(bg="black")

title_label = tk.Label(root, text="AI Voice Assistant", font=("Arial", 16), fg="white", bg="black")
title_label.pack(pady=10)

status_label = tk.Label(root, text="Click Start and say 'hi'", fg="cyan", bg="black")
status_label.pack()

start_button = tk.Button(root, text="Start Listening", command=start_thread, bg="green", fg="white")
start_button.pack(pady=10)

output_box = ScrolledText(root, height=20, bg="black", fg="white")
output_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

root.mainloop()