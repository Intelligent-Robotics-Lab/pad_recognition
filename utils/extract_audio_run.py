import os
from moviepy import VideoFileClip
from utils.helpers import extract_all_meld_audio, extract_audio_ffmpeg

meld_root = r"C:\Users\carte\emotion-recognition-hri\data\MELD.Raw"

extract_all_meld_audio(meld_root)

print("Audio extraction complete!")