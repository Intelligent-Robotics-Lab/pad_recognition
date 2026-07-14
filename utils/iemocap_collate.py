import torch

def iemocap_collate(batch):

    texts = []
    audios = []
    sample_rates = []
    pads = []

    for item in batch:
        texts.append(item["text"])
        audios.append(item["audio"])
        sample_rates.append(item["sample_rate"])
        pads.append(item["pad"])

    return {
        "text": texts,
        "audio": audios,
        "sample_rate": sample_rates,
        "pad": torch.stack(pads)
    }