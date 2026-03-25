from transformers import AutoTokenizer, AutoModel, pipeline
import torch

# Returns shape of (T_text, d_model) in our case (T_ 768)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load in the pretrained models
embedding_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("bert-base-uncased").to(device)  # Use GPU if available, otherwise CPU

embedding_model.eval()  # Disable dropout and other features

def extract_text_features(text):
    # Ensure input is a single string
    if not isinstance(text, str):
        text = str(text)

    # Tokenize text
    inputs = embedding_tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=False
    ).to(device)

    # Forward pass
    outputs = embedding_model(**inputs)
    token_embeddings = outputs.last_hidden_state  # [B, T, 768]

    # Remove batch dimension [T, 768]

    token_embeddings = token_embeddings.squeeze(0)

    # Light, stable normalization across hidden_dim
    mean = token_embeddings.mean(dim=0, keepdim=True)
    std = token_embeddings.std(dim=0, keepdim=True) + 1e-6
    token_embeddings = (token_embeddings - mean) / std

    return token_embeddings.cpu()

# Optional prep function, changed functionality but still used this name throughout so kept it
def prepare_text_features(text):
    return extract_text_features(text)