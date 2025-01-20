import time
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from sklearn.metrics import precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class NERModelComparison:
    def __init__(self, model_names: list, tokenizer_names: list, dataset_path: str):
        """
        Initialize the NER model comparison.

        Args:
            model_names (list): List of model names to compare (e.g., XLM-Roberta, DistilBERT, mBERT).
            tokenizer_names (list): List of tokenizer names corresponding to the models.
            dataset_path (str): Path to the dataset (in CoNLL format).
        """
        self.model_names = model_names
        self.tokenizer_names = tokenizer_names
        self.dataset_path = dataset_path
        self.best_model = None
        self.best_metrics = {}
        self.label2id = None  # Initialize label2id as None

    def load_and_prepare_data(self):
        """Loads and prepares data from CoNLL file."""
        df = self.load_conll_data(self.dataset_path)
        
        # Split the data into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
        
        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_names[0])
        
        # Initialize label2id mapping based on the unique labels in the train set
        unique_labels = set(tag for tags in dataset['train']['ner_tags'] for tag in tags)
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Initialize id2label mapping
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Tokenize and align labels
        tokenized_datasets = dataset.map(
            lambda examples, idx: self.tokenize_and_align_labels(examples, tokenizer),
            batched=True,
            with_indices=True
        )

        return tokenized_datasets

    def load_conll_data(self, filepath: str):
        """Loads CoNLL-formatted data into a Pandas DataFrame."""
        sentences, labels = [], []
        sentence, label = [], []
        
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip() == '':  # Empty line marks the end of a sentence
                    if sentence:
                        sentences.append(sentence)
                        labels.append(label)
                        sentence, label = [], []
                else:
                    token, tag = line.strip().split()
                    sentence.append(token)
                    label.append(tag)
        
        return pd.DataFrame({"tokens": sentences, "ner_tags": labels})

    def tokenize_and_align_labels(self, examples, tokenizer):
        """Tokenizes the inputs and aligns labels."""
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding='max_length', is_split_into_words=True, max_length=128)
        labels = []
        
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id.get(label[word_idx], -1))  # Ensure a valid label
                else:
                    label_ids.append(-100)  # Ignore subsequent tokens of the same word
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs['labels'] = labels
        return tokenized_inputs

    def compute_metrics(self, p):
        """Computes metrics for evaluation."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Filter out the -100 labels from the ground truth and predictions
        true_labels = [[self.id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.id2label[pred] for (pred, label) in zip(prediction, label) if label != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten the lists of true labels and predictions for evaluation
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        flat_true_predictions = [item for sublist in true_predictions for item in sublist]
        
        return {
            "precision": precision_score(flat_true_labels, flat_true_predictions, average="weighted"),
            "recall": recall_score(flat_true_labels, flat_true_predictions, average="weighted"),
            "f1": f1_score(flat_true_labels, flat_true_predictions, average="weighted")
        }

    def train_and_evaluate(self, model_name: str, tokenizer_name: str, tokenized_datasets):
        """Trains and evaluates a given model with specific conditions for each model."""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Create label mappings
        unique_labels = set(tag for tags in tokenized_datasets['train']['ner_tags'] for tag in tags)
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Initialize model
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(unique_labels), id2label=self.id2label, label2id=self.label2id)
        
        data_collator = DataCollatorForTokenClassification(tokenizer)
        
        # Define model-specific training conditions
        if model_name == "xlm-roberta-base":
            batch_size = 16
            learning_rate = 3e-5
            epochs = 4
        elif model_name == "distilbert-base-uncased":
            batch_size = 8
            learning_rate = 2e-5
            epochs = 3
        elif model_name == "bert-base-multilingual-cased":
            batch_size = 16
            learning_rate = 2e-5
            epochs = 3
        else:
            batch_size = 8
            learning_rate = 2e-5
            epochs = 3

        training_args = TrainingArguments(
            output_dir=f"./{model_name}_model",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_strategy="epoch",
            save_total_limit=2,
            no_cuda=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        # Evaluate the model
        eval_results = trainer.evaluate()
        
        return eval_results, training_time

    def compare_models(self):
        """Compares models based on evaluation metrics."""
        tokenized_datasets = self.load_and_prepare_data()
        
        best_model = None
        best_f1 = 0
        best_model_name = None
        best_training_time = float("inf")
        
        for model_name, tokenizer_name in zip(self.model_names, self.tokenizer_names):
            print(f"Training and evaluating {model_name}...")
            eval_results, training_time = self.train_and_evaluate(model_name, tokenizer_name, tokenized_datasets)
            
            print(f"{model_name} Evaluation Results: {eval_results}")
            print(f"{model_name} Training Time: {training_time} seconds")
            
            # Select the best model based on F1 score and training time
            if eval_results['eval_f1'] > best_f1 and training_time < best_training_time:
                best_f1 = eval_results['eval_f1']
                best_training_time = training_time
                best_model = model_name
                best_model_name = model_name
        
        self.best_model = best_model_name
        self.best_metrics = {
            'f1': best_f1,
            'training_time': best_training_time
        }
        print(f"Best Model: {self.best_model}")
        print(f"Best Metrics: {self.best_metrics}")

