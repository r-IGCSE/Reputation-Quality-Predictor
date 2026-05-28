from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("chiraglol/r-igcse-reputation-quality-predictor")
model = AutoModelForSequenceClassification.from_pretrained(
    "chiraglol/r-igcse-reputation-quality-predictor",
    num_labels=1
).to(device)

def predict_quality(formatted_messages: str) -> float:
    inputs = tokenizer(formatted_messages, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        quality_score = outputs.logits.item() 

        return quality_score

def get_token_count(formatted_messages: str) -> int:
    inputs = tokenizer(formatted_messages, return_tensors="pt")
    return inputs.input_ids.shape[1]