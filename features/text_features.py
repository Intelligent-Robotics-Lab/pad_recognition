import torch
from transformers import AutoTokenizer, AutoModel, pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using roberta-large model (stronger than BERT-base)
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModel.from_pretrained("roberta-large").to(device)
model.eval()

def extract_text_features(text: str) -> torch.tensor:
    # Returns: Tensor [T, 1024]
    if not isinstance(text, str):
        text = str(text)

    # Extract token-level embeddings
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.squeeze(0).cpu()

# Removable with new functionality but keeping due to naming consistency and potential future use
def prepare_text_features(text):
    return extract_text_features(text)