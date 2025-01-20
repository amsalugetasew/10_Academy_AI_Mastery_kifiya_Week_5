import lime
from lime.lime_text import LimeTextExplainer
import numpy as np
import torch

class NERModelInterpretabilityWithLIME(NERModelComparison):
    def __init__(self, model_names, tokenizer_names, dataset_path):
        super().__init__(model_names, tokenizer_names, dataset_path)

    def explain_with_lime(self, model_name: str, tokenizer_name: str, tokenized_datasets):
        """Use LIME to explain model predictions"""
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # LIME text explainer
        explainer = LimeTextExplainer(class_names=[str(i) for i in range(len(self.label2id))])
        
        # Define a prediction function for LIME
        def predict_fn(texts):
            inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
            outputs = model(**inputs).logits
            return outputs.detach().numpy()
        
        # Example input text for explanation
        example_text = tokenized_datasets['train']['tokens'][0]  # Take the first tokenized sentence

        # Explain the prediction for the example
        explanation = explainer.explain_instance(example_text, predict_fn, num_features=10)
        
        # Visualize the explanation
        explanation.show_in_notebook(text=True)
        return explanation

