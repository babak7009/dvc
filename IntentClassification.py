# from dvclive import Live
# live = Live()

from datasets import load_dataset

dataset = load_dataset('banking77')
dataset


# %matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from dvclive.huggingface import DvcLiveCallback
from dvclive import Live



train_texts = [item["text"] for item in dataset["train"]][:100]
train_labels = [item["label"] for item in dataset["train"]][:100]

test_texts = [item["text"] for item in dataset["test"]]
test_labels = [item["label"] for item in dataset["test"]]

label_counter = Counter(train_labels)
label_names = dataset["train"].features["label"].names
label_frequencies = {label_names[label]: [label_counter[label]] for label in label_counter}

# df = pd.DataFrame.from_dict(label_frequencies, orient="index", columns=["frequency"])
# df = df.sort_values("frequency", ascending=False)



train_texts, dev_texts, train_labels, dev_labels = train_test_split(train_texts, train_labels, test_size=0.1, shuffle=True, random_state=1)

print("Train:", len(train_texts))
print("Dev:", len(dev_texts))
print("Test:", len(test_texts))



class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc
    }


model_ids = ["prajjwal1/bert-tiny", "prajjwal1/bert-mini"]
             # ,"prajjwal1/bert-small", "prajjwal1/bert-medium",
             # "albert-base-v2", "albert-large-v2", "bert-base-uncased"]

accuracies = []
live = Live()

for model_id in model_ids:
    
    print(f"*** {model_id} ***")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(label_names))

    train_texts_encoded = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
    dev_texts_encoded = tokenizer(dev_texts, padding=True, truncation=True, return_tensors="pt")
    test_texts_encoded = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
    
    train_dataset = ClassificationDataset(train_texts_encoded, train_labels)
    dev_dataset = ClassificationDataset(dev_texts_encoded, dev_labels)
    test_dataset = ClassificationDataset(test_texts_encoded, test_labels)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=int(len(train_dataset)/16),
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=10,
        load_best_model_at_end=True,
        no_cuda=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    metric_name = "eval_accuracy"
    trainer.add_callback(DvcLiveCallback())
    trainer.train()
    test_results = trainer.evaluate(test_dataset)
    accuracies.append(test_results[metric_name])
    live.log(metric_name, test_results[metric_name])
live.next_step()