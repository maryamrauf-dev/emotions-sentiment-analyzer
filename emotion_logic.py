import re
import emoji
import os
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification, pipeline

def load_classifier():
    """
    Loads the emotion classification model.
    Uses a fine-tuned XLNet model if available, otherwise falls back to a 
    standard emotion model.
    """
    MODEL_PATH = "my_finetuned_model"
    BASE_MODEL = "xlnet-base-cased"
    
    try:
        if os.path.exists(MODEL_PATH):
            model = XLNetForSequenceClassification.from_pretrained(MODEL_PATH)
            tokenizer = XLNetTokenizer.from_pretrained(BASE_MODEL)
        else:
            model = XLNetForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=4,
                id2label={0: "fear", 1: "anger", 2: "joy", 3: "sadness"},
                label2id={"fear": 0, "anger": 1, "joy": 2, "sadness": 3}
            )
            tokenizer = XLNetTokenizer.from_pretrained(BASE_MODEL)
        
        return pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception:
        # Fallback to a simpler model if the above fails
        return pipeline("text_classification", model="bhadresh-savani/bert-base-uncased-emotion")

def preprocess_text(text: str):
    """
    Cleans the input text by removing emojis and extra spaces.
    """
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    # Remove special characters except basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_single_text(classifier, text: str):
    """
    Analyzes a single piece of text and returns sorted results.
    """
    clean_text = preprocess_text(text)
    results = classifier(clean_text, top_k=None)
    # Sort results by confidence score (highest first)
    return sorted(results, key=lambda x: x['score'], reverse=True)
