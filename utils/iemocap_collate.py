import torch

def iemocap_collate(batch):

    texts = [item["text"] for item in batch]

    audios = [item["audio"] for item in batch]

    pads = torch.stack(
        [item["pad"] for item in batch]
    )

    return {
        "text": texts,
        "audio": audios,
        "pad": pads
    }