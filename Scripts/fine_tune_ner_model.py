# # # Import necessary libraries
# # import tensorflow as tf
# # from transformers import TFAutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
# # from datasets import load_dataset, load_metric
# # class Fine_Tune_NER_Model:
# #     def __init__(self):
# #         self.df = {}
# #         # Install necessary libraries
# #         # !pip install transformers datasets seqeval tensorflow

# #         # Load the dataset in CoNLL format
# #         self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# #         dataset = load_dataset("conll2003")  # Replace with your Amharic dataset in CoNLL format if applicable

# #         # Load the tokenizer and model
# #         model_checkpoint = "xlm-roberta-base"  # Replace with "bert-tiny-amharic" or "afroxlmr" if applicable
# #         tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# #         # Define label names
# #         label_list = dataset["train"].features["ner_tags"].feature.names

# #     # Tokenize the dataset and align labels
# #     def tokenize_and_align_labels(self,examples):
# #         tokenized_inputs = self.tokenizer(
# #             examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True, max_length=128
# #         )
        
# #         labels = []
# #         for i, label in enumerate(examples["ner_tags"]):
# #             word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word IDs
# #             previous_word_idx = None
# #             label_ids = []
# #             for word_idx in word_ids:
# #                 if word_idx is None:
# #                     label_ids.append(-100)
# #                 elif word_idx != previous_word_idx:
# #                     label_ids.append(label[word_idx])
# #                 else:
# #                     label_ids.append(-100)
# #                 previous_word_idx = word_idx
# #             labels.append(label_ids)
        
# #         tokenized_inputs["labels"] = labels
# #         return tokenized_inputs

# #     tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# #     # Prepare data for TensorFlow
# #     data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")

# #     train_dataset = tokenized_datasets["train"].to_tf_dataset(
# #         columns=["attention_mask", "input_ids", "labels"],
# #         shuffle=True,
# #         batch_size=16,
# #         collate_fn=data_collator,
# #     )

# #     val_dataset = tokenized_datasets["validation"].to_tf_dataset(
# #         columns=["attention_mask", "input_ids", "labels"],
# #         shuffle=False,
# #         batch_size=16,
# #         collate_fn=data_collator,
# #     )

# #     # Load the pre-trained model
# #     model = TFAutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

# #     # Define optimizer, loss, and metrics
# #     optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# #     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
# #     metric = load_metric("seqeval")

# #     # Compile the model
# #     model.compile(optimizer=optimizer, loss=loss)

# #     # Define a custom callback for evaluation
# #     class NERMetricCallback(tf.keras.callbacks.Callback):
# #         def on_epoch_end(self, epoch, logs=None):
# #             predictions, labels = [], []
# #             for batch in val_dataset:
# #                 logits = model.predict(batch)["logits"]
# #                 predictions.extend(tf.argmax(logits, axis=-1).numpy())
# #                 labels.extend(batch["labels"].numpy())
            
# #             # Flatten and align labels for evaluation
# #             true_predictions = [
# #                 [label_list[p] for (p, l) in zip(pred, label) if l != -100]
# #                 for pred, label in zip(predictions, labels)
# #             ]
# #             true_labels = [
# #                 [label_list[l] for (p, l) in zip(pred, label) if l != -100]
# #                 for pred, label in zip(predictions, labels)
# #             ]
# #             results = metric.compute(predictions=true_predictions, references=true_labels)
# #             print(f"\nEpoch {epoch + 1} - Evaluation: {results}")

# #     # Train the model
# #     model.fit(
# #         train_dataset,
# #         validation_data=val_dataset,
# #         epochs=3,
# #         callbacks=[NERMetricCallback()]
# #     )

# #     # Save the fine-tuned model
# #     model.save_pretrained("fine_tuned_ner_model")
# #     tokenizer.save_pretrained("fine_tuned_ner_model")




# # Install necessary libraries
# # !pip install transformers datasets seqeval tensorflow

# # Import necessary libraries
# import tensorflow as tf
# from transformers import TFAutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
# from datasets import load_dataset, load_metric

# class NERFineTuner:
#     def __init__(self, model_checkpoint, dataset_name, batch_size=16, max_length=128, learning_rate=5e-5):
#         self.model_checkpoint = model_checkpoint
#         self.dataset_name = dataset_name
#         self.batch_size = batch_size
#         self.max_length = max_length
#         self.learning_rate = learning_rate
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
#         self.metric = load_metric("seqeval")
#         self.model = None
#         self.label_list = None
#         self.train_dataset = None
#         self.val_dataset = None

#     def load_and_preprocess_data(self):
#         # Load dataset
#         dataset = load_dataset(self.dataset_name)
#         self.label_list = dataset["train"].features["ner_tags"].feature.names

#         # Tokenize and align labels
#         def tokenize_and_align_labels(examples):
#             tokenized_inputs = self.tokenizer(
#                 examples["tokens"], truncation=True, padding="max_length",
#                 is_split_into_words=True, max_length=self.max_length
#             )

#             labels = []
#             for i, label in enumerate(examples["ner_tags"]):
#                 word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word IDs
#                 previous_word_idx = None
#                 label_ids = []
#                 for word_idx in word_ids:
#                     if word_idx is None:
#                         label_ids.append(-100)
#                     elif word_idx != previous_word_idx:
#                         label_ids.append(label[word_idx])
#                     else:
#                         label_ids.append(-100)
#                     previous_word_idx = word_idx
#                 labels.append(label_ids)

#             tokenized_inputs["labels"] = labels
#             return tokenized_inputs

#         tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

#         # Prepare TensorFlow datasets
#         data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, return_tensors="tf")
#         self.train_dataset = tokenized_datasets["train"].to_tf_dataset(
#             columns=["attention_mask", "input_ids", "labels"],
#             shuffle=True,
#             batch_size=self.batch_size,
#             collate_fn=data_collator,
#         )
#         self.val_dataset = tokenized_datasets["validation"].to_tf_dataset(
#             columns=["attention_mask", "input_ids", "labels"],
#             shuffle=False,
#             batch_size=self.batch_size,
#             collate_fn=data_collator,
#         )

#     def load_model(self):
#         self.model = TFAutoModelForTokenClassification.from_pretrained(
#             self.model_checkpoint, num_labels=len(self.label_list)
#         )

#     def compile_model(self):
#         optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
#         loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
#         self.model.compile(optimizer=optimizer, loss=loss)

#     def train_model(self, epochs=3):
#         # Define custom callback for evaluation
#         class NERMetricCallback(tf.keras.callbacks.Callback):
#             def __init__(self, model, val_dataset, label_list, metric):
#                 super().__init__()
#                 self.model = model
#                 self.val_dataset = val_dataset
#                 self.label_list = label_list
#                 self.metric = metric

#             def on_epoch_end(self, epoch, logs=None):
#                 predictions, labels = [], []
#                 for batch in self.val_dataset:
#                     logits = self.model.predict(batch)["logits"]
#                     predictions.extend(tf.argmax(logits, axis=-1).numpy())
#                     labels.extend(batch["labels"].numpy())

#                 # Flatten and align labels for evaluation
#                 true_predictions = [
#                     [self.label_list[p] for (p, l) in zip(pred, label) if l != -100]
#                     for pred, label in zip(predictions, labels)
#                 ]
#                 true_labels = [
#                     [self.label_list[l] for (p, l) in zip(pred, label) if l != -100]
#                     for pred, label in zip(predictions, labels)
#                 ]
#                 results = self.metric.compute(predictions=true_predictions, references=true_labels)
#                 print(f"\nEpoch {epoch + 1} - Evaluation: {results}")

#         # Train the model
#         self.model.fit(
#             self.train_dataset,
#             validation_data=self.val_dataset,
#             epochs=epochs,
#             callbacks=[NERMetricCallback(self.model, self.val_dataset, self.label_list, self.metric)],
#         )

#     def save_model(self, output_dir="fine_tuned_ner_model"):
#         self.model.save_pretrained(output_dir)
#         self.tokenizer.save_pretrained(output_dir)
#         print(f"Model saved to {output_dir}")


# import tensorflow as tf
# from transformers import TFAutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
# from datasets import load_dataset
# import evaluate

import tensorflow as tf
from transformers import TFAutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
import evaluate  # Use the evaluate library instead of datasets.load_metric
from datasets import load_dataset

class NERFineTuner:
    def __init__(self, model_checkpoint, dataset_name, batch_size=16, max_length=128, learning_rate=5e-5):
        self.model_checkpoint = model_checkpoint  # Now accepts the model_checkpoint parameter
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)  # Use model_checkpoint here
        
        # Load evaluation metric using evaluate library
        self.metric = evaluate.load("seqeval")
        
        self.model = None
        self.label_list = None
        self.train_dataset = None
        self.val_dataset = None

    def load_and_preprocess_data(self, data):
        # Load dataset
        # dataset = load_dataset(self.dataset_name)
        # if isinstance(data, list):
        #     data = {"train": data}  # Wrap list into a dictionary with 'train' key for consistency
        dataset = data
        self.label_list = dataset["train"].features["ner_tags"].feature.names

        # Tokenize and align labels
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
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

        tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

        # Prepare TensorFlow datasets
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, return_tensors="tf")
        self.train_dataset = tokenized_datasets["train"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "labels"],
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=data_collator,
        )
        self.val_dataset = tokenized_datasets["validation"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "labels"],
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=data_collator,
        )

    def load_model(self):
        self.model = TFAutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint, num_labels=len(self.label_list)
        )

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.model.compile(optimizer=optimizer, loss=loss)

    def train_model(self, epochs=3):
        # Define custom callback for evaluation
        class NERMetricCallback(tf.keras.callbacks.Callback):
            def __init__(self, model, val_dataset, label_list, metric):
                super().__init__()
                self.model = model
                self.val_dataset = val_dataset
                self.label_list = label_list
                self.metric = metric

            def on_epoch_end(self, epoch, logs=None):
                predictions, labels = [], []
                for batch in self.val_dataset:
                    logits = self.model.predict(batch)["logits"]
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
                print(f"\nEpoch {epoch + 1} - Evaluation: {results}")

        # Train the model
        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=[NERMetricCallback(self.model, self.val_dataset, self.label_list, self.metric)],
        )

    def save_model(self, output_dir="fine_tuned_ner_model"):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")




