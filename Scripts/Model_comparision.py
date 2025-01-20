# Install necessary libraries
# !pip install transformers datasets seqeval tensorflow evaluate

# Import libraries
import tensorflow as tf
from transformers import TFAutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
import evaluate  # Import the evaluate library instead of datasets
from datasets import load_dataset

class NERModelComparison:
    def __init__(self, dataset_name, batch_size=16, max_length=128, learning_rate=5e-5):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_length = max_length
        # self.dataset = load_dataset("json", data_files=self.dataset_name)
        # self.dataset = Dataset.from_dict({"tokens": [d["tokens"] for d in data], "ner_tags": [d["ner_tags"] for d in data]})
        self.dataset = load_dataset("conll", data_files=self.dataset_name)
        # if isinstance(self.dataset_name, list):
        #     self.dataset_name = "".join(self.dataset_name)  # Convert list to string
        # self.dataset = load_dataset(self.dataset_name)
        self.label_list = self.dataset["train"].features["ner_tags"].feature.names
        
        # Replace load_metric with evaluate.load
        self.metric = evaluate.load("seqeval")  # Use evaluate to load the metric
        self.results = {}

    def tokenize_and_align_labels(self, examples, tokenizer):
        """Tokenize inputs and align labels for NER."""
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, padding="max_length",
            is_split_into_words=True, max_length=self.max_length
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word IDs
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def preprocess_data(self, model_checkpoint):
        """Preprocess data for the given model."""
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        tokenized_datasets = self.dataset.map(
            lambda x: self.tokenize_and_align_labels(x, tokenizer), batched=True
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")

        train_dataset = tokenized_datasets["train"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "labels"],
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=data_collator,
        )
        val_dataset = tokenized_datasets["validation"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "labels"],
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=data_collator,
        )
        return tokenizer, train_dataset, val_dataset

    def evaluate_model(self, model, val_dataset):
        """Evaluate the model on the validation set."""
        predictions, labels = [], []
        for batch in val_dataset:
            logits = model.predict(batch)["logits"]
            predictions.extend(tf.argmax(logits, axis=-1).numpy())
            labels.extend(batch["labels"].numpy())

        # Flatten and align labels for evaluation
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return results

    def train_and_evaluate(self, model_checkpoint, epochs=3):
        """Train and evaluate a single model."""
        print(f"\nTraining model: {model_checkpoint}")
        tokenizer, train_dataset, val_dataset = self.preprocess_data(model_checkpoint)
        
        # Load the model
        model = TFAutoModelForTokenClassification.from_pretrained(
            model_checkpoint, num_labels=len(self.label_list)
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        model.compile(optimizer=optimizer, loss=loss)

        # Train the model
        model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

        # Evaluate the model
        results = self.evaluate_model(model, val_dataset)
        print(f"Evaluation Results for {model_checkpoint}: {results}")

        # Store results
        self.results[model_checkpoint] = {
            "results": results,
            "model": model,
            "tokenizer": tokenizer
        }

    def compare_models(self, model_checkpoints, epochs=3):
        """Fine-tune and compare multiple models."""
        for model_checkpoint in model_checkpoints:
            self.train_and_evaluate(model_checkpoint, epochs)

    def save_best_model(self, output_dir="best_ner_model"):
        """Select and save the best-performing model."""
        best_model_checkpoint = max(self.results, key=lambda x: self.results[x]["results"]["overall_f1"])
        best_model = self.results[best_model_checkpoint]["model"]
        best_tokenizer = self.results[best_model_checkpoint]["tokenizer"]

        # Save the model and tokenizer
        best_model.save_pretrained(output_dir)
        best_tokenizer.save_pretrained(output_dir)
        print(f"Best model ({best_model_checkpoint}) saved to {output_dir}")



