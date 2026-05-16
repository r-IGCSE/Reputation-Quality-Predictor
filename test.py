from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("./output/final")
model = AutoModelForSequenceClassification.from_pretrained(
    "./output/final", 
    num_labels=1
).to(device)

special_tokens_dict = {'additional_special_tokens': ['[A]', '[R]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

data = json.load(open("test.json", "r"))

true_labels = []
predicted_labels = []
results = []

for item in data:
    text = item["text"]
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # get number of tokens in the input
    num_tokens = inputs["input_ids"].shape[1]
    
    print("=======================================================")
    print(f"Input: {text}")

    print(f"Number of tokens in the input: {num_tokens}")

    with torch.no_grad():
        outputs = model(**inputs)
        quality_score = outputs.logits.item()
        
    label = item["label"]
    print(f"Inferred Quality Score: {quality_score}")
    print(f"Expected Score: {label}")
    print(f"Deviation: {quality_score - label}")
    true_labels.append(label)
    predicted_labels.append(quality_score)
    
    results.append({
        "input": text,
        "predicted_score": quality_score,
        "true_score": label,
        "deviation": quality_score - label
    })

# to be used for reference later
with open("inference_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("=======================================================")
print("Evaluation Metrics:")
print(f"Mean Absolute Error: {mean_absolute_error(true_labels, predicted_labels)}")
print(f"Mean Squared Error: {mean_squared_error(true_labels, predicted_labels)}")
print(f"R^2 Score: {r2_score(true_labels, predicted_labels)}")
