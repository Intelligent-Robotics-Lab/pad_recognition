import torch

def iemocap_collate(batch):

    texts = []
    audios = []
    sample_rates = []
    video_feats = []
    pads = []

    for item in batch:
        texts.append(item["text"])
        audios.append(item["audio"])
        sample_rates.append(item["sample_rate"])
        video_feats.append(item["video_feats"])
        pads.append(item["pad"])

    return {
        "text": texts,
        "audio": audios,
        "sample_rate": sample_rates,
        "video_feats": video_feats,
        "pad": torch.stack(pads)
    }