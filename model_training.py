from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification, EarlyStoppingCallback
from datasets import load_from_disk
import numpy as np
import torch
import os
from sklearn.metrics import precision_score, recall_score, f1_score


# Function to align labels with tokens
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            new_labels.append(label)
    return new_labels


# Function to tokenize and align entity labels
def tokenize_and_align_entity_labels(dataset_samples):
    tokenized_outputs = tokenizer(
        dataset_samples["tokens"], truncation=True, is_split_into_words=True
    )
    all_entity_labels = dataset_samples["fine_ner_tags"]
    aligned_labels = []

    for index, entity_labels in enumerate(all_entity_labels):
        word_indices = tokenized_outputs.word_ids(index)
        aligned_labels.append(align_labels_with_tokens(entity_labels, word_indices))

    tokenized_outputs["labels"] = aligned_labels
    return tokenized_outputs


# Function to calculate performance metrics for NER
def calculate_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    true_labels = [label[label != -100] for label in labels]
    true_preds = [pred[label != -100] for pred, label in zip(preds, labels)]

    all_labels = np.concatenate(true_labels)
    all_preds = np.concatenate(true_preds)

    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {"precision": precision, "recall": recall, "f1": f1}


# Function to calculate custom loss with CrossEntropyLoss
def calculate_loss(model, batch_data):
    target_labels = batch_data.get("labels")
    model_outputs = model(**batch_data)
    predicted_logits = model_outputs.get('logits')

    # Ignore labels with value -100
    target_labels[target_labels == -100] = 0

    # Compute loss using CrossEntropyLoss with class weights
    num_labels = model.config.num_labels
    loss_function = torch.nn.CrossEntropyLoss()

    computed_loss = loss_function(predicted_logits.view(-1, num_labels), target_labels.view(-1))

    return computed_loss


def train_model(model, tokenizer, train_set, val_set, data_collator, lr=2e-5, epochs=3, early_stopping_patience=3):
    os.makedirs('./models/logs', exist_ok=True)

    args = TrainingArguments(
        "./models/logs",
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=8,  # based on available memory
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        weight_decay=0.01,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        compute_metrics=calculate_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # Start training the model
    trainer.train()
    trainer.save_model()

    # Save the model and tokenizer
    model_save_path = os.path.join("./saved_model", 'model')
    tokenizer_save_path = os.path.join("./saved_model", 'tokenizer')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)


if __name__ == '__main__':
    save_dir = "./saved_model"

    # Load tokenizer and model from pre-trained BERT
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased")

    train_loaded = load_from_disk('./data/train_data')
    val_loaded = load_from_disk('./data/val_data')

    # Tokenize and align the datasets
    tokenized_train = train_loaded.map(
        tokenize_and_align_entity_labels,
        batched=True
    )
    tokenized_val = val_loaded.map(
        tokenize_and_align_entity_labels,
        batched=True
    )

    # Create a data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_model(model, tokenizer, tokenized_train, tokenized_val, data_collator, epochs=5)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(os.path.join(save_dir, 'model'))
        tokenizer.save_pretrained(os.path.join(save_dir, 'tokenizer'))
