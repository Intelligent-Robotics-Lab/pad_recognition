from pathlib import Path
import re


class IEMOCAPParser:
    """Parser converts raw IEMOCAP data into usable rows for the CSV file"""

    def __init__(self, root):
        self.root = Path(root)

    def parse(self):
        rows = []

        for session in range(1, 6):

            session_path = self.root / f"Session{session}"

            trans_dir = session_path / "dialog" / "transcriptions"
            emo_dir = session_path / "dialog" / "EmoEvaluation"

            wav_root = session_path / "sentences" / "wav"

            avi_root = session_path / "dialog" / "avi" / "DivX"


            for trans_file in trans_dir.glob("*.txt"):

                if trans_file.name.startswith("._"):
                    continue

                conversation = trans_file.stem

                emo_file = emo_dir / trans_file.name

                if not emo_file.exists():
                    continue

                transcripts = self._parse_transcript(trans_file)
                emotions = self._parse_emotions(emo_file)

                for utt_id, text in transcripts.items():

                    if utt_id not in emotions:
                        continue

                    wav_path = self._find_audio(wav_root, utt_id)

                    if wav_path is None:
                        continue

                    video_path = self._find_video(avi_root, conversation)

                    if video_path is None:
                        continue

                    row = {
                        "utterance_id": utt_id,
                        "session": session,
                        "conversation": conversation,

                        "text": text,

                        "audio_path": str(
                            wav_path.relative_to(self.root.parent)
                        ),

                        "video_path": str(
                            video_path.relative_to(self.root.parent)
                        ),

                        "start_time": emotions[utt_id]["start_time"],
                        "end_time": emotions[utt_id]["end_time"],

                        "emotion": emotions[utt_id]["emotion"],

                        "valence": emotions[utt_id]["valence"],
                        "arousal": emotions[utt_id]["arousal"],
                        "dominance": emotions[utt_id]["dominance"],
                    }

                    rows.append(row)

        return rows

    def _parse_transcript(self, file):
        data = {}

        with open(file, "r", errors="ignore") as f:

            for line in f:

                match = re.match(
                    r"(Ses\d+[MF]_\w+_\w+\d+)\s+\[.*\]:\s+(.*)",
                    line
                )

                if match:
                    utt_id = match.group(1)
                    text = match.group(2)

                    data[utt_id] = text.strip()

        return data

    def _parse_emotions(self, file):
        data = {}

        with open(file, "r", errors="ignore") as f:

            for line in f:

                match = re.match(
                    r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+"
                    r"(Ses\d+[MF]_\w+_\w+\d+)\s+(\w+)\s+"
                    r"\[([\d.]+),\s*([\d.]+),\s*([\d.]+)\]",
                    line
                )

                if match:

                    utt_id = match.group(3)

                    data[utt_id] = {
                        "emotion": match.group(4),

                        "start_time": float(match.group(1)),
                        "end_time": float(match.group(2)),

                        # Normalize to a [-1, 1] scale instead of [1, 5] scale
                        "valence": max(-1, min(1, (float(match.group(5)) - 3) / 2)),
                        "arousal": max(-1, min(1, (float(match.group(6)) - 3) / 2)),
                        "dominance": max(-1, min(1, (float(match.group(7)) - 3) / 2)),
                    }

        return data


    def _find_audio(self, root, utt_id):

        for wav in root.rglob(f"{utt_id}.wav"):
            return wav

        return None

    def _find_video(self, root, conversation):

        video = root / f"{conversation}.avi"

        if video.exists():
            return video

        return None