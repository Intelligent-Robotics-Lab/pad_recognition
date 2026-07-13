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

                    row = {
                        "utterance_id": utt_id,
                        "session": session,
                        "conversation": conversation,

                        "text": text,

                        "audio_path": str(
                            wav_path.relative_to(self.root.parent)
                        ),

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
                    r"(Ses\d+_\w+_\w+\d+)\s+\[.*\]:\s+(.*)",
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
                    r"\[.*\]\s+(Ses\d+_\w+_\w+\d+)\s+(\w+)\s+\[(.*),(.*),(.*)\]",
                    line
                )

                if match:
                    utt_id = match.group(1)

                    data[utt_id] = {
                        "emotion": match.group(2),

                        # convert from a [1, 5] scale to a [-1, 1]
                        "valence":
                            (float(match.group(3)) - 3) / 2,

                        "arousal":
                            (float(match.group(4)) - 3) / 2,

                        "dominance":
                            (float(match.group(5)) - 3) / 2,
                    }

        return data


    def _find_audio(self, root, utt_id):

        for wav in root.rglob(f"{utt_id}.wav"):
            return wav

        return None