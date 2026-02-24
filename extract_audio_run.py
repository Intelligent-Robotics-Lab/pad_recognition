# extract_audio_run.py
import os
from moviepy import VideoFileClip
from utils.helpers import extract_all_meld_audio, extract_audio_ffmpeg

meld_root = r"C:\Users\carte\emotion-recognition-hri\data\MELD.Raw"

# # Run the extraction
# extract_all_meld_audio(meld_root)

# # Test the extraction on a single file
# mp4_path = r"C:\Users\carte\emotion-recognition-hri\data\MELD.Raw\train\dia1_utt1.mp4"
# wav_path = r"C:\Users\carte\emotion-recognition-hri\data\MELD.Raw\train\dia1_utt1.wav"

extract_all_meld_audio(meld_root)

print("Audio extraction complete!")