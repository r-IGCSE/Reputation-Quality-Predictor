import json

from mongo import Message, ReputationData, ReputationQueue
from ocr import process_image, get_file_extension
from predictor import predict_quality, get_token_count

user_map = {
    "Asker": "[A]",
    "Responder": "[R]"
}

# question_trim and image_trim are the maximum length of question content and image ocr result to keep
def format_messages(messages: list[Message], image_trim: int | None = None, question_trim: int | None = None) -> str:
    formatted = ""
    if not messages:
        return ""
    
    # if last message is by the Responder, remove it
    if messages[-1].author == "Responder":
        messages.pop()
        
    if not messages:
        return ""
    
    # if the last message is by the Asker, remove it as well
    if messages[-1].author == "Asker":
        messages.pop()
        
    if not messages:
        return ""
    
    formatted += "Question: "

    # group and join consecutive messages by the same author
    current_author: str = ""
    current_messages: list[str] = []
    
    grouped_messages: list[tuple[str, list[str]]] = []
    
    for message in messages:
        for attachment_url in message.attachmentUrls:
            file_extension = get_file_extension(attachment_url)
            if file_extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"]:
                ocr_result = process_image(attachment_url)
                if image_trim is not None and ocr_result:
                    message.content += f" [IMAGE OCR]: {ocr_result[:image_trim]}"
                elif ocr_result:
                    message.content += f" [IMAGE OCR]: {ocr_result}"
            else:
                message.content += f" [ATTACHMENT]: {file_extension} file"
            
        if message.author != current_author:
            if current_messages:
                grouped_messages.append((current_author, current_messages))
            current_author = message.author
            current_messages = [message.content]
        else:
            current_messages.append(message.content)
    
    if not grouped_messages:
        return ""
    
    if current_messages:
        if grouped_messages[-1][0] == current_author:
            grouped_messages[-1][1].extend(current_messages)
        else:
            grouped_messages.append((current_author, current_messages))
            
    # get the latest message group by the Responder and all the other message groups
    latest_responder_index: int = -1
    for i in range(len(grouped_messages)):
        if grouped_messages[i][0] == "Responder":
            latest_responder_index = i
            break
        
    # remove item at index latest_responder_index from the array, and get all other items in the array in separate arrays
    other_message_groups: list[tuple[str, list[str]]] = []
    for i in range(len(grouped_messages)):
        if i != latest_responder_index:
            other_message_groups.append(grouped_messages[i])
            
    latest_responder_group: tuple[str, list[str]] = grouped_messages[latest_responder_index]
    
    for author, msgs in other_message_groups:
        formatted += f"{user_map[author]} {'. '.join(msgs)} "
    
    if question_trim is not None:
        formatted = formatted[:question_trim]
        
    formatted += f"[SEP] Answer: {'. '.join(latest_responder_group[1])}"
        
    return formatted.strip()
    
inference_results = []

def main():
    for item in ReputationQueue.objects(): # type: ignore
        print(f"Processing item with id {item.id} and reputationDataObjectId {item.reputationDataObjectId}")
        formatted_input = format_messages(item.input)
        
        print(f"Formatted messages: {formatted_input}")
        token_count = get_token_count(formatted_input)
        decrement = 0
        
        while token_count > 512:
            decrement += 25
            print(f"Token count: {token_count}")
            print(f"Token count exceeds 512, truncating image content (if any) or question content")
            formatted_input = format_messages(item.input, image_trim=150-decrement, question_trim=400-decrement)
            print(f"Truncated formatted messages: {formatted_input}")
            token_count = get_token_count(formatted_input)
            
        quality_score = predict_quality(formatted_input)
        
        inference_results.append({
            "input": formatted_input,
            "predicted_score": quality_score
        })
    
    json.dump(inference_results, open("inference_results.json", "w"), indent=4)

if __name__ == "__main__":
    main()