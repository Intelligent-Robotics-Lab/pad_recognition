from transformers import AutoTokenizer, AutoModel, pipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
embedding_model = AutoModel.from_pretrained("roberta-large").to(device)
embedding_model.eval()

def extract_text_features(text):
    if not isinstance(text, str):
        text = str(text)

    inputs = embedding_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = embedding_model(**inputs)

    return outputs.last_hidden_state.squeeze(0).cpu()

# Removable with new functionality but keeping due to naming consistency and potential future use
def prepare_text_features(text):
    return extract_text_features(text)