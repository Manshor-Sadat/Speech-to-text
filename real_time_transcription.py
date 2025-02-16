import pyaudio
import wave
import datetime
import queue
import threading
import warnings
import os
import time
from faster_whisper import WhisperModel

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load Faster-Whisper (Optimized for Speed)
model = WhisperModel("small", device="cpu", compute_type="int8", num_workers=2)

# Audio settings
FORMAT = pyaudio.paInt16  
CHANNELS = 1  
RATE = 16000  
CHUNK = 4096  
RECORD_SECONDS = 1  # Process every second for true real-time
AUDIO_FOLDER = "recordings"  

# Ensure the recordings folder exists
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Queue for real-time transcription processing
audio_queue = queue.Queue()

# Function to find and use the correct microphone
def get_correct_microphone():
    audio = pyaudio.PyAudio()
    correct_device = None
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        if "MacBook" in dev["name"] or "Built-in" in dev["name"]:  # Ensure using built-in mic
            correct_device = i
            break
    return correct_device if correct_device is not None else 0  # Default to device 0 if none found

# Get correct microphone device
input_device_index = get_correct_microphone()
print(f"✅ Using Input Device: {input_device_index}")

# Function to process transcription in real time
def transcribe_audio():
    full_transcript = []
    
    while True:
        audio_file = audio_queue.get()
        if audio_file is None:
            break  # Stop thread when recording ends
        
        # Ensure the file exists before processing
        time.sleep(0.2)  # Short delay to ensure file is written
        if os.path.exists(audio_file):
            segments, _ = model.transcribe(audio_file)
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            for segment in segments:
                text = f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}"
                print(f"[{timestamp}] {segment.text}")  # Show live transcription
                full_transcript.append(text)

            os.remove(audio_file)  # Delete only after successful transcription
        else:
            print(f"⚠️ Skipping missing file: {audio_file}")

    # Save final transcript after the call ends
    with open("zoom_transcription.txt", "w") as f:
        for line in full_transcript:
            f.write(line + "\n")
    
    print("\n✅ Transcription saved as 'zoom_transcription.txt'")

# Start transcription processing in a separate thread
transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
transcription_thread.start()

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK, 
                    input_device_index=input_device_index)

print("🎤 Real-Time Transcription Started... Press Ctrl+C to stop.")
start_time = datetime.datetime.now()

try:
    while True:
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except OSError:
                print("⚠️ Buffer overflow, skipping this chunk...")
                continue

        # Save the latest audio chunk
        audio_filename = os.path.join(AUDIO_FOLDER, f"chunk_{datetime.datetime.now().strftime('%H%M%S')}.wav")
        with wave.open(audio_filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        # Add only the new chunk to the queue
        audio_queue.put(audio_filename)

except KeyboardInterrupt:
    print("\n🛑 Stopping real-time transcription...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    audio_queue.put(None)  
    transcription_thread.join(timeout=1)  # Timeout prevents hanging

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n📌 Total Call Duration: {duration:.2f} seconds")
    exit(0)  # Cleanly exit
