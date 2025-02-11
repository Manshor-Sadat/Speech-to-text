import pyaudio
import numpy as np
import whisper

# Load Whisper Model
model = whisper.load_model("small")

# Audio Configuration
chunk = 4096
sample_format = pyaudio.paInt16
channels = 1
sample_rate = 16000

# Supported languages
languages = ["en", "es", "fr", "de", "zh", "ar"]
print("Supported languages:", languages)
language_code = input("Select a language (default is 'en'): ") or "en"

# Initialize PyAudio
p = pyaudio.PyAudio()
print("Listening to system audio (Press Ctrl+C to stop)")

stream = p.open(format=sample_format,
                 channels=channels,
                 rate=sample_rate,
                 input=True,
                 frames_per_buffer=chunk)

audio_buffer = []
buffer_duration = 5  # 5-second buffer

try:
    while True:
        # Read audio chunk
        data = stream.read(chunk, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_buffer.extend(audio_chunk)

        # Process audio buffer when it reaches the target duration
        if len(audio_buffer) >= sample_rate * buffer_duration:
            print("Processing...")
            # Convert to numpy array and write to temporary WAV file
            audio_array = np.array(audio_buffer)
            
            # Transcribe using Whisper
            result = model.transcribe(audio_array, language=language_code)

            # Print real-time transcription result
            if result["text"].strip():
                print(f"You said: {result['text']}")

            # Clear the buffer for the next segment
            audio_buffer = []

except KeyboardInterrupt:
    print("Stopped by user.")
    stream.stop_stream()
    stream.close()
    p.terminate()
