import torch

def meld_collate(batch):

    texts = []
    audios = []
    sample_rates = []
    video_paths = []
    start_times = []
    end_times = []
    pads = []

    for item in batch:
        texts.append(item["text"])
        audios.append(item["audio"])
        sample_rates.append(item["sample_rate"])
        video_paths.append(item["video_path"])
        start_times.append(item["start_time"])
        end_times.append(item["end_time"])
        pads.append(item["pad"])

    return {
        "text": texts,
        "audio": audios,
        "sample_rate": sample_rates,
        "video_path": video_paths,
        "start_time": start_times,
        "end_time": end_times,
        "pad": torch.stack(pads)
    }
