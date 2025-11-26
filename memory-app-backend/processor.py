import os
from pyannote.audio import Pipeline
import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment

class AudioProcessor:
    def __init__(self):
        # Initialize pyannote pipeline
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")
        
        print("Loading pyannote pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        print("✓ Pyannote loaded")
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        print("✓ Speech recognizer ready")
    
    def convert_to_wav(self, audio_path):
        """Convert M4A to WAV for better compatibility"""
        wav_path = str(audio_path).replace('.m4a', '.wav')
        
        if not os.path.exists(wav_path):
            print(f"Converting {audio_path} to WAV...")
            audio = AudioSegment.from_file(audio_path, format="m4a")
            audio.export(wav_path, format="wav")
            print(f"✓ Converted to {wav_path}")
        
        return wav_path
    
    def diarize(self, audio_path):
        """
        Perform speaker diarization on audio file
        Returns: dict with speaker segments
        """
        # Convert to WAV first
        wav_path = self.convert_to_wav(audio_path)
        
        print(f"Running diarization on {wav_path}...")
        
        diarization = self.diarization_pipeline(wav_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "duration": turn.end - turn.start
            })
        
        print(f"✓ Found {len(set(s['speaker'] for s in segments))} speaker(s)")
        return segments
    
    def transcribe_segment(self, audio_path, start, end):
        """
        Transcribe a specific segment of audio
        """
        wav_path = self.convert_to_wav(audio_path)
        
        # Load audio
        audio = AudioSegment.from_wav(wav_path)
        
        # Extract segment (times are in seconds, pydub uses milliseconds)
        segment = audio[int(start * 1000):int(end * 1000)]
        
        # Save temporary segment
        temp_path = wav_path.replace('.wav', f'_temp_{start}_{end}.wav')
        segment.export(temp_path, format="wav")
        
        try:
            # Transcribe the segment
            with sr.AudioFile(temp_path) as source:
                audio_data = self.recognizer.record(source)
            
            text = self.recognizer.recognize_whisper(audio_data)
            
            # Clean up temp file
            os.remove(temp_path)
            
            return text.strip()
        
        except Exception as e:
            print(f"  ✗ Segment transcription failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return ""
    
    def process(self, audio_path):
        """
        Complete processing: diarization + transcription per segment
        """
        print(f"\n{'='*60}")
        print(f"Processing: {Path(audio_path).name}")
        print(f"{'='*60}")
        
        # Diarize
        segments = self.diarize(audio_path)
        
        # Transcribe each segment
        print(f"Transcribing {len(segments)} segment(s)...")
        
        labeled_segments = []
        for i, seg in enumerate(segments):
            print(f"  [{i+1}/{len(segments)}] Transcribing {seg['speaker']} ({seg['start']:.1f}s - {seg['end']:.1f}s)...")
            
            text = self.transcribe_segment(audio_path, seg['start'], seg['end'])
            
            labeled_segments.append({
                "speaker": seg['speaker'],
                "start": seg['start'],
                "end": seg['end'],
                "duration": seg['duration'],
                "text": text
            })
            
            print(f"    ✓ \"{text}\"")
        
        # Combine results
        result = {
            "file": str(audio_path),
            "segments": labeled_segments,
            "num_speakers": len(set(s['speaker'] for s in segments))
        }
        
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"Speakers detected: {result['num_speakers']}")
        print(f"\nTranscript with speaker labels:")
        for seg in labeled_segments:
            print(f"  [{seg['start']:.1f}s] {seg['speaker']}: {seg['text']}")
        print(f"{'='*60}\n")
        
        return result


# Test function
def test_processor():
    """Test with one of the uploaded files"""
    processor = AudioProcessor()
    
    # Get first file from uploads
    uploads_dir = Path("uploads")
    audio_files = list(uploads_dir.glob("*.m4a"))
    
    if not audio_files:
        print("No audio files found in uploads/")
        return
    
    test_file = audio_files[0]
    print(f"Testing with: {test_file}")
    
    result = processor.process(str(test_file))
    
    # Print the full dictionary
    print("\n" + "="*60)
    print("FULL RESULT DICTIONARY:")
    print("="*60)
    import json
    print(json.dumps(result, indent=2))
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    test_processor()