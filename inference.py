from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("./output/final")
model = AutoModelForSequenceClassification.from_pretrained(
    "./output/final", 
    num_labels=1
).to(device)

text = "" # input text
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# giv number of tokens in the input
num_tokens = inputs["input_ids"].shape[1]

print(f"Number of tokens in the input: {num_tokens}")

with torch.no_grad():
    outputs = model(**inputs)
    quality_score = outputs.logits.item() 

print(f"Quality Score: {quality_score}")