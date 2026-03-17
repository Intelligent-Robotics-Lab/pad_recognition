from transformers import AutoTokenizer, AutoModel, pipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load in the pretrained models
embedding_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("bert-base-uncased").to(device)  # Use GPU if available, otherwise CPU

embedding_model.eval()  # Disable dropout and other features

def extract_text_features(text):
    # Handle single string input
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    with torch.no_grad():
        # Tokenize text
        inputs = embedding_tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(device)
        # Run BERT pretrained model to get token embeddings
        outputs = embedding_model(**inputs)
        # Keep the full sequence of token embeddings (go back to mean pooling if struggling)
        token_embeddings = outputs.last_hidden_state.detach()  # [B, T, 768]

        # Simple per utterance normalization
        mean = token_embeddings.mean(dim=1,keepdim=True) # Mean over tokens
        std = token_embeddings.std(dim=1, keepdim=True) + 1e-6 # Std over tokens and avoid divide by 0
        token_embeddings = (token_embeddings - mean) / std

    return token_embeddings, inputs["attention_mask"] # Return attention mask too for training

# Optional prep function, changed functionality but still used this name throughout so kept it
def prepare_text_features(text):
    return extract_text_features(text)