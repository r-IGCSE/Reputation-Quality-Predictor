# Reputation Quality Predictor
This branch (`model`) contains the code for training and testing the fine-tuned model used for this project.

The base model used for this project is `distilbert-base-uncased`

The model is given an input in the following format:
```
Question: [A] text1 [R] text2 [A] text3 [IMAGE OCR]: image-text1 [SEP] Answer: text4
```
where `[A]` stands for the (question) Asker, `[R]` stands for the Responder
`text4` is a text sent by the responder to which the asker has replied to with a `thank you` or similar.

The model outputs a score between 0 to 1, depending on the quality of the interaction and helpfulness of the Responder.