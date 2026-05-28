# Reputation Quality Predictor
The sole purpose of this project is to act as a stand-alone server for the [r/IGCSE Bot](https://github.com/r-IGCSE/r-igcse-bot) to detect quality of user responses sent in the [r/IGCSE Discord Server](https://discord.gg/igcse).

## What is reputation?
In all subject channels on the r/IGCSE Discord Server, users gain reputation (or "rep") by being thanked by other users for answering questions, clearing doubts, taking study sessions etc.

## What problem does this solve?
The main aim behind a quality factor for each reputation is to ensure much fairer results in the Helper of the Month ("HOTM") competition held in r/IGCSE every month.

As of May 2025, the current format is by voting. However, we believe that voting is not a real measure of how helpful a person has been to others, and is rather a popularity contest.

Our next approach was to use reputation awarded to user in a month, the user with the most reputation awarded wins. While that certainly is a better approach than voting, we strongly believe that quantity on its own is not enough a factor for us to declare HOTM results on.

This is where quality factor comes in. Quality factor is calculated for each rep awarded, ranging from 0 to 1.<br>
A score of 0 indicating that the rep holds no value and will not be counted towards future statistics and HOTM.

Quality factor mainly depends on the following factors:
- Original question (asked by the "rep giver")
- Reply/replies (by the "rep receiver")
- Previous reps given to the same reply (by rep giver or other users)

Images in original question and subsequent reply/replies are roughly converted to text using OCR, however due to DistilBERT limitations, may be trimmed to a certain length.

However, to ensure fairness across channels which may have a significantly less amount of active users, there is also a multiplier which may be manually set per channel.

This is planned to be implemented on or before 1st July, 2026.

## How is the quality factor determined?
The quality factor is determined by a fine-tuned DistilBERT model, the fine-tuned model can be found on [HuggingFace](), the training script can be found in the `model` branch on this repo.

The data for training this model is artificially written/generated with reference to data from the r/IGCSE Discord Server. No real messages on the server have been used to train this model and will never be used to train this model.

## Why not use a low parameter LLM?
Cost and speed.

r/IGCSE being a volunteer-run community has very limited resources.<br>
We average about 1000+ reps awarded a day during peak months, while using a LLM might be easier, the cost will be much higher.<br>
Given the limited resources we have, using a LLM is not feasible.


# Getting Started
## Notice
Please note that this project is mainly built to be used only with the [r/IGCSE Bot](https://github.com/r-IGCSE/r-igcse-bot), all future updates will follow any and all changes in the original bot, which may include breaking changes in the future.

## TODO

