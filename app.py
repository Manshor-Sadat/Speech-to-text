# import pyaudio
# import numpy as np
# import whisper

# # Load Whisper Model
# model = whisper.load_model("small")

# # Audio Configuration
# chunk = 4096
# sample_format = pyaudio.paInt16
# channels = 1
# sample_rate = 16000

# # Supported languages
# languages = ["en"]
# print("Supported languages:", languages)
# language_code = input("Select a language (default is 'en'): ") or "en"

# # Initialize PyAudio
# p = pyaudio.PyAudio()
# print("Listening to system audio (Press Ctrl+C to stop)")
# stream = p.open(format=sample_format,
#                  channels=channels,
#                  rate=sample_rate,
#                  input=True,
#                  frames_per_buffer=chunk)
# audio_buffer = []
# buffer_duration = 5  

# try:
#     while True:
    
#         data = stream.read(chunk, exception_on_overflow=False)
#         audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
#         audio_buffer.extend(audio_chunk)
#         if len(audio_buffer) >= sample_rate * buffer_duration:
#             print("Processing...")
#             audio_array = np.array(audio_buffer)
#             result = model.transcribe(audio_array, language=language_code)

#             if result["text"].strip():
#                 print(f"You said: {result['text']}")
#             audio_buffer = []

# except KeyboardInterrupt:
#     print("Stopped by user.")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

import assemblyai as aai
import whisper
import sounddevice as sd
import numpy as np
import io
import wave
import time
import threading
from queue import Queue
from typing import Dict, List
import gc  # For garbage collection

# 🗝️ Initialize AssemblyAI
aai.settings.api_key = "ada3df25b909471ca405dd86fc221940"
transcriber = aai.Transcriber()

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration=10):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.recording = False
        self.audio_chunks = []
        self.current_chunk = []
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.lock = threading.Lock()
        
    def callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        if self.recording:
            with self.lock:
                self.current_chunk.extend(indata.flatten())
                if len(self.current_chunk) >= self.chunk_samples:
                    chunk_array = np.array(self.current_chunk[:self.chunk_samples])
                    self.audio_chunks.append(chunk_array)
                    self.current_chunk = self.current_chunk[self.chunk_samples:]
    
    def start(self):
        self.recording = True
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self.callback,
            dtype=np.float32,
            blocksize=4096  # Smaller blocksize for better memory handling
        )
        self.stream.start()
        
    def stop(self):
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Handle any remaining audio in current_chunk
        with self.lock:
            if self.current_chunk:
                final_chunk = np.array(self.current_chunk)
                if len(final_chunk) > 0:
                    self.audio_chunks.append(final_chunk)
        
    def get_audio(self):
        if not self.audio_chunks:
            return None
        
        # Concatenate chunks and clear the list to free memory
        audio = np.concatenate(self.audio_chunks)
        self.audio_chunks = []  # Clear chunks to free memory
        gc.collect()  # Force garbage collection
        
        # Normalize audio
        if len(audio) > 0 and np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio

def create_wav_buffer(audio_data, sample_rate=16000):
    try:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        wav_buffer.seek(0)
        return wav_buffer
    except Exception as e:
        print(f"Error creating WAV buffer: {e}")
        return None

def transcribe_with_whisper(audio_array):
    try:
        # Use tiny model for minimal memory usage
        model = whisper.load_model("tiny")
        
        # Ensure audio is float32 and memory efficient
        audio_array = audio_array.astype(np.float32, copy=False)
        
        result = model.transcribe(
            audio_array,
            language="en",
            temperature=0.0,
            fp16=False
        )
        
        print("\n📝 Whisper Transcription:")
        print(result["text"])
        
        # Clear memory
        del model
        gc.collect()
        
        return result["text"]
    except Exception as e:
        print(f"❌ Whisper transcription error: {str(e)}")
        return None

def diarize_with_assemblyai(audio_stream):
    try:
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            punctuate=True
        )
        
        print("\nProcessing with AssemblyAI...")
        transcript = transcriber.transcribe(audio_stream, config=config)
        
        if not transcript.utterances:
            print("⚠️ No speech detected.")
            return None
        
        print("\n🔊 Speaker Diarization Results:")
        conversation = []
        
        for utterance in transcript.utterances:
            speaker = f"person{utterance.speaker}"
            print(f"👤 {speaker}: {utterance.text}")
            conversation.append({
                'speaker': speaker,
                'text': utterance.text,
                'start': utterance.start,
                'end': utterance.end
            })
        
        return conversation
    except Exception as e:
        print(f"❌ AssemblyAI error: {str(e)}")
        return None

def main():
    print("\n🎯 Voice Recognition System")
    print("Press Enter to start recording")
    print("Press Enter again to stop recording")
    
    try:
        input("Ready? Press Enter to begin...")
        recorder = AudioRecorder(chunk_duration=5)  # Shorter chunks
        recorder.start()
        print("🎤 Recording... Press Enter to stop")
        input()
        
        print("Stopping recording...")
        recorder.stop()
        print("Processing audio...")
        
        audio_data = recorder.get_audio()
        if audio_data is None:
            print("No audio recorded!")
            return
        
        # Process with Whisper
        print("Transcribing with Whisper...")
        transcribe_with_whisper(audio_data)
        
        # Process with AssemblyAI
        print("Creating WAV buffer...")
        wav_buffer = create_wav_buffer(audio_data)
        if wav_buffer:
            diarize_with_assemblyai(wav_buffer)
        
        # Clean up
        del audio_data
        gc.collect()
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nSession completed.")

if __name__ == "__main__":
    main()