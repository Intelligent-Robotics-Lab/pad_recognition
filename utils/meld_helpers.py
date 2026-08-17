import os
import subprocess


def extract_audio_from_mp4(mp4_path, save_path, target_sr=16000):
    from moviepy import VideoFileClip

    if os.path.exists(save_path):
        print(f"Skipped (already exists): {save_path}")
        return save_path

    clip = VideoFileClip(mp4_path)

    clip.audio.write_audiofile(save_path, fps=target_sr, verbose=False, logger=None)
    clip.close()
    print(f"Saved: {save_path}")
    return save_path

# Audio extraction using ffmpeg for better performance and reliability compared to moviepy
def extract_audio_ffmpeg(mp4_path, wav_path, target_sr=16000):
    if os.path.exists(wav_path):
        print(f"Skipped (already exists): {wav_path}")
        return wav_path

    ffmpeg_path = r"C:\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

    cmd = [
        ffmpeg_path,
        "-i", mp4_path,         # input file
        "-vn",                  # ignore video
        "-ac", "1",             # convert to mono
        "-ar", str(target_sr),  # sampling rate
        "-y",                   # overwrite if exists
        wav_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Saved: {wav_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting {mp4_path}: {e}")

    return wav_path

# Walk through MELD train/dev/test folders and extract audio for all .mp4 files
def extract_all_meld_audio(root_dir):
    for split in ["train", "dev", "test"]:
        folder = os.path.join(root_dir, split)

        if not os.path.exists(folder):
            print(f"Folder does not exist, skipping: {folder}")
            continue

        print(f"\nProcessing {split} folder...")
        mp4_files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
        print(f"Found {len(mp4_files)} MP4 files in {folder}")  # debug

        for f in mp4_files:
            mp4_path = os.path.abspath(os.path.join(folder, f))
            wav_path = os.path.abspath(os.path.join(folder, f.replace(".mp4", ".wav")))

            if os.path.exists(wav_path):
                print("Already exists, skipping:", wav_path)
                continue

            try:
                extract_audio_ffmpeg(mp4_path, wav_path)
            except Exception as e:
                print(f"Error processing {mp4_path}: {e}")
