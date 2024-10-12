from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


def predict_entities(text, model, tokenizer):
    inputs = tokenizer(text.split(), return_tensors="pt", is_split_into_words=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predictions and decode them
    predictions = torch.argmax(outputs.logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Convert predicted labels to words and modify the output text
    modified_text = []
    for token, prediction in zip(tokens, predictions[0].numpy()):
        if token.startswith('##'):
            # Handle subword tokens by combining them with the previous token
            token = token[2:]
            if modified_text:
                modified_text[-1] += token
        else:
            if prediction != 0:
                modified_token = f"**{token}**"
                modified_text.append(modified_token)
            else:
                modified_text.append(token)

    result_text = tokenizer.convert_tokens_to_string(modified_text)

    return result_text


if __name__ == '__main__':
    model_path = "./saved_model/model"
    tokenizer_path = "./saved_model/tokenizer"

    ner_model = AutoModelForTokenClassification.from_pretrained(model_path)
    ner_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    input_text = "I love the Rocky Mountains and Mount Everest."

    # Perform inference
    modified_text = predict_entities(input_text, ner_model, ner_tokenizer)

    print(f"Modified text with highlighted entities:", modified_text)
