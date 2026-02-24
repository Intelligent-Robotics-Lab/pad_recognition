from transformers import AutoTokenizer, AutoModel, pipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once
embedding_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("bert-base-uncased").to(device) # GPU if available, else CPU
embedding_model.eval()  # Set to evaluation mode

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1)

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0 if torch.cuda.is_available() else -1)

def extract_text_features(text):
    with torch.no_grad():
        inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = embedding_model(**inputs)
        bert_embedding = outputs.last_hidden_state.mean(dim=1).detach().flatten()

    # Sentiment scalar
    sent = sentiment_model(text)[0]
    sentiment_value = sent["score"] if sent["label"] == "POSITIVE" else -sent["score"]
    sentiment_tensor = torch.tensor([sentiment_value], dtype=torch.float32, device=device)

    # Emotion distribution vector (switched from only using top score)
    emotions = emotion_model(text)[0]
    emotion_scores = torch.tensor(
        [e["score"] for e in emotions], dtype=torch.float32, device=device)

    # Stack raw features into single vector (not final representation)
    features = torch.cat([bert_embedding, sentiment_tensor, emotion_scores])

    return features

# This is the final prep function for here
def prepare_text_features(text):
    return extract_text_features(text)