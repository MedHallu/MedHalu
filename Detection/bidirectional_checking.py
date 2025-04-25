import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

class EntailmentAnalyzer:
    def __init__(self):
        # Load RoBERTa model and tokenizer for NLI
        self.model_name = "roberta-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def check_entailment(self, premise, hypothesis):
        # Handle None or empty strings
        if not premise or not hypothesis or pd.isna(premise) or pd.isna(hypothesis):
            return 0.0

        # Convert to string if not already
        premise = str(premise)
        hypothesis = str(hypothesis)

        # Prepare input
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            
        # Get probability for entailment (index 2 in RoBERTa-MNLI)
        entailment_prob = predictions[0][2].item()
        return entailment_prob

    def get_bidirectional_score(self, text1, text2):
        """Calculate bidirectional entailment score between two texts"""
        forward_score = self.check_entailment(text1, text2)
        backward_score = self.check_entailment(text2, text1)
        return min(forward_score, backward_score)

    def classify_similarity(self, score, threshold=0.5):
        """Classify texts as same or different based on bidirectional entailment score"""
        if pd.isna(score):
            return "unknown"
        return "same" if score >= threshold else "different"

def process_dataframe(input_path, output_path, threshold=0.5):
    """Process the dataframe and add entailment analysis columns"""
    try:
        # Verify input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Initialize analyzer
        analyzer = EntailmentAnalyzer()
        
        # Read the input dataframe
        print(f"Reading file: {input_path}")
        df = pd.read_csv(input_path)
        
        # Verify required columns exist
        required_columns = ['least_similar_answer', 'ground_truth']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print("Processing rows...")
        # Initialize new columns
        df['bidirectional_entailment_score'] = 0.0
        df['similarity_classification'] = ''
        
        # Process each row
        total_rows = len(df)
        for idx, row in df.iterrows():
            if idx % 10 == 0:  # Progress update every 10 rows
                print(f"Processing row {idx}/{total_rows}")
                
            # Calculate bidirectional entailment score
            score = analyzer.get_bidirectional_score(
                row['least_similar_answer'], 
                row['ground_truth']
            )
            
            # Store score and classification
            df.at[idx, 'bidirectional_entailment_score'] = score
            df.at[idx, 'similarity_classification'] = analyzer.classify_similarity(
                score, 
                threshold
            )
        
        # Create output directory if it doesn't exist
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed dataframe
        print(f"Saving results to: {output_path}")
        df.to_csv(output_path, index=False)
        print("Processing completed successfully!")
        
        return df

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Define your input and output paths
    input_file = ""  # Replace with your input file path
    output_file = ""  # Replace with your output file path
    
    try:
        processed_df = process_dataframe(input_file, output_file)
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Failed to process file: {str(e)}")