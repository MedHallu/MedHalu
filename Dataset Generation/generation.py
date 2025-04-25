import os
import pandas as pd
from tqdm import tqdm
import argparse
import json
import numpy as np
import openai
from openai import OpenAI
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import re
import torch
from transformers import pipeline
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Tuple
from enum import Enum
import textgrad as tg
multiprocessing.set_start_method('spawn', force=True)

# ============ Model Type Enums and Configs ============
class ModelType(Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class ModelConfig:
    model_type: ModelType
    model_id: str
    temperature: float = 0.3
    max_tokens: int = 64
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    top_p: float = 0.9

# ============ Hyperparameters and Configuration ============
NUM_GENERATIONS = 5
BATCH_SIZE = 9000
GENERATOR_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
# TEMPERATURE = 0.8
GENERATOR_TEMPERATURE = 0.8
DISCRIMINATOR_TEMPERATURE = 0.3
TOP_P = 0.95
MAX_NEW_TOKENS = 512
OUTPUT_FILE = " "
CHECKPOINT_FILE = " "

GENERATOR_CONFIG = ModelConfig(
    model_type=ModelType.HUGGINGFACE,
    model_id=GENERATOR_MODEL_ID,
    temperature=GENERATOR_TEMPERATURE,
    max_tokens=MAX_NEW_TOKENS,
    top_p=TOP_P
)

# Discriminator configurations
DISCRIMINATOR_CONFIGS = [
    ModelConfig(
        model_type=ModelType.OPENAI, 
        model_id="gpt-4o-mini",
        temperature=DISCRIMINATOR_TEMPERATURE
    ),
    ModelConfig(
        model_type=ModelType.HUGGINGFACE, 
        model_id="google/gemma-2-2b-it",
        temperature=DISCRIMINATOR_TEMPERATURE,
        top_p=TOP_P,
        max_tokens = 4,
    ),
    ModelConfig(
        model_type=ModelType.HUGGINGFACE, 
        model_id="Qwen/Qwen2.5-3B-Instruct",
        temperature=DISCRIMINATOR_TEMPERATURE,
        top_p=TOP_P,
        max_tokens = 4
    )
]

# ============ System Prompts ============
def load_prompt(file_path):
    """Load system prompt from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            system_prompt = file.read().strip()
        return system_prompt
    except FileNotFoundError:
        raise FileNotFoundError(f"System prompt file not found at: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading system prompt file: {str(e)}")

# ============ LLM Wrapper Class ============
class LLMWrapper:
    def __init__(self, config: ModelConfig, openai_client=None):
        self.config = config
        self.client = openai_client
        self.pipe = None
        self.terminators = None
        
        if config.model_type == ModelType.HUGGINGFACE:
            self._setup_hf_pipeline()
    
    def _setup_hf_pipeline(self):
        try:
            self.pipe = pipeline(
                'text-generation',
                model=self.config.model_id,
                tokenizer=self.config.model_id,
                model_kwargs={"torch_dtype": self.config.torch_dtype},
                device_map="auto"
            )
            if "llama" in self.config.model_id.lower():
                self.terminators = [
                    self.pipe.tokenizer.eos_token_id,
                    self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
            elif "gemma" in self.config.model_id.lower():
                self.terminators = [
                    self.pipe.tokenizer.eos_token_id,
                    self.pipe.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                ]
            else:
                self.terminators = [self.pipe.tokenizer.eos_token_id]
        except Exception as e:
            print(f"Error setting up HF pipeline for {self.config.model_id}: {e}")
            raise

    def generate(self, messages: Union[str, List[Dict[str, str]]], max_new_tokens: int = None) -> str:
        try:
            if self.config.model_type == ModelType.OPENAI:
                return self._generate_openai(messages)
            else:
                return self._generate_hf(messages, max_new_tokens)
        except Exception as e:
            print(f"Error in generate for {self.config.model_id}: {e}")
            return ""

    def _generate_openai(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                max_tokens=self.config.max_tokens,
                n=1,
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in OpenAI generation: {e}")
            return ""

    def _generate_hf(self, messages: str, max_new_tokens: int = None) -> str:
        try:
            tokens = max_new_tokens if max_new_tokens else self.config.max_tokens
            outputs = self.pipe(
                messages,
                max_new_tokens=tokens,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                return_full_text=False,
            )
            
            # HuggingFace pipeline returns a list of dictionaries with 'generated_text' key
            if isinstance(outputs, list) and len(outputs) > 0:
                return outputs[0]['generated_text'].strip()
            return ""
            
        except Exception as e:
            print(f"Error in HuggingFace generation: {e}")
            return ""

# ============ Utility Functions ============
def calculate_semantic_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Calculate semantic similarity between two texts using sentence transformers."""
    try:
        if not text1 or not text2:
            return 0.0
        
        embeddings1 = model.encode(text1, convert_to_tensor=True)
        embeddings2 = model.encode(text2, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return float(cosine_similarity.item())
    except Exception as e:
        print(f"Error in semantic similarity calculation: {e}")
        return 0.0
    
def create_prompt(question: str, option1: str, option2: str, justification: str) -> str:
    """Create a prompt for the discriminator model."""
    prompt = f"""
    Question: {question}
    Option 1: {option1}
    Option 2: {option2} + {justification}
    
    Return just the answer, for example: "Option 1" or "Option 2". Don't return anything else, just the answer.
    """
    return prompt

def get_sections(text: str) -> Tuple[str, str, str, str, str]:
    """Extract different sections from the generated text."""
    try:
        pattern = r'#([^#]+)#:\s*([^#]+)'
        matches = re.findall(pattern, text)
        sections = {}
        for label, content in matches:
            sections[label.strip()] = content.strip()
        
        return (sections.get('Question', ''),
                sections.get('Knowledge', ''),
                sections.get('Ground truth answer', ''),
                sections.get('Hallucinated Answer', ''),
                sections.get('Justification of Hallucinated answer', ''))
    except Exception as e:
        print(f"Error parsing sections: {e}")
        return '', '', '', '', ''

def determine_difficulty(discriminator_results: List[bool]) -> DifficultyLevel:
    """Determine the difficulty level based on which discriminators were fooled."""
    num_fooled = sum(discriminator_results)
    
    if num_fooled == len(discriminator_results):
        return DifficultyLevel.HARD
    elif num_fooled > 1:
        return DifficultyLevel.MEDIUM
    else:
        return DifficultyLevel.EASY

def create_empty_results_df() -> pd.DataFrame:
    """Create an empty DataFrame with the required columns including per-discriminator results."""
    columns = ['question', 'knowledge', 'ground_truth']
    
    for i in range(NUM_GENERATIONS):
        columns.extend([
            f'hallucinated_answer_{i+1}', 
            f'justification_{i+1}'
        ])
        for j, config in enumerate(DISCRIMINATOR_CONFIGS):
            columns.append(f'fooled_discriminator_{j+1}_{i+1}')
        columns.append(f'difficulty_level_{i+1}')
    
    columns.append('least_similar_answer')
    columns.append('final_difficulty_level')
    return pd.DataFrame(columns=columns)

def save_checkpoint(df: pd.DataFrame, filename: str = CHECKPOINT_FILE):
    """Save the current state of the DataFrame to a checkpoint file."""
    try:
        df.to_csv(filename, index=False)
        print(f"Checkpoint saved to {filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        
# ============ Model Initialization ============
def initialize_models(openai_api_key: str = "", hf_model_id: str = GENERATOR_MODEL_ID):
    """Initialize all required models."""
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize sentence transformer model
        sent_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize generator (HuggingFace wrapper)
        # generator_config = ModelConfig(
        #     model_type=ModelType.HUGGINGFACE,
        #     model_id=hf_model_id,
        #     temperature=TEMPERATURE,
        #     top_p=TOP_P
        # )
        
        generator_wrapper = LLMWrapper(GENERATOR_CONFIG)
        
        # Initialize discriminator wrappers
        discriminator_wrappers = []
        for config in DISCRIMINATOR_CONFIGS:
            if config.model_type == ModelType.OPENAI:
                wrapper = LLMWrapper(config, openai_client=openai_client)
            else:
                wrapper = LLMWrapper(config)
            discriminator_wrappers.append(wrapper)
        
        return generator_wrapper, discriminator_wrappers, sent_model
    
    except Exception as e:
        print(f"Error initializing models: {e}")
        raise

def setup_textgrad_optimization(failed_text: str, question: str, knowledge: str) -> tg.Variable:
    """Setup TextGrad optimization for failed hallucination."""
    # Set GPT-4 as the backward engine for gradient computation
    tg.set_backward_engine("gpt-4o-mini", override=True)
    
    # Create variable for the failed attempt
    failed_attempt = tg.Variable(
        failed_text,
        requires_grad=True,
        role_description="hallucinated answer that needs improvement"
    )
    
    # Create context variables
    context = tg.Variable(
        f"Question: {question}\nKnowledge Context: {knowledge}",
        requires_grad=False,
        role_description="context for hallucination generation"
    )
    
    return failed_attempt, context

def create_textgrad_loss(question: str) -> tg.TextLoss:
    """Create a TextGrad loss function for hallucination evaluation."""
    evaluation_instruction = tg.Variable(
        f"""Evaluate the given hallucinated answer for this question: {question}
        Consider:
        1. Plausibility given the context, is it on-topic and relevant?
        2. Internal consistency, does the answer make sense and is it logically coherent?
        3. Use of domain-specific terminology
        4. How easy is it for the discriminator to detect the answer as a hallucination?
        
        Provide specific feedback on how to improve the answer to make it more convincing.
        Be concise and direct.""",
        requires_grad=False,
        role_description="evaluation instructions"
    )
    
    return tg.TextLoss(evaluation_instruction)

def improve_hallucination_with_textgrad(
    failed_text: str,
    question: str,
    knowledge: str
) -> str:
    """Improve failed hallucination using TextGrad optimization."""
    try:
        # Setup TextGrad variables
        failed_attempt, context = setup_textgrad_optimization(failed_text, question, knowledge)
        
        # Create optimizer
        optimizer = tg.TGD(parameters=[failed_attempt])
        
        # Create loss function
        loss_fn = create_textgrad_loss(question)
        
        # Optimization loop
        for _ in range(3):  # Multiple optimization steps
            # Compute loss
            loss = loss_fn(failed_attempt)
            
            # Backward pass
            loss.backward()
            
            # Update the text
            optimizer.step()
            
            # Clear gradients
            optimizer.zero_grad()
        
        return failed_attempt.value
        
    except Exception as e:
        print(f"Error in TextGrad optimization: {e}")
        return failed_text
    
def check_hallu_multiple(
    answer: str, 
    discriminator_wrappers: List[LLMWrapper]
) -> Tuple[List[bool], Optional[str], Optional[str]]:
    """Check if the hallucinated answer can fool multiple discriminators."""
    print("\nChecking hallucination against multiple discriminators...")
    try:
        question, knowledge, ground_truth_answer, hallucinated_answer, justification = get_sections(answer)
        
        if not all([hallucinated_answer, justification]):
            print("Missing required sections in generated text")
            return [False] * len(discriminator_wrappers), None, None
        
        option1 = ground_truth_answer
        option2 = hallucinated_answer
        prompt = create_prompt(question, option1, option2, justification)
        
        results = []
        for i, wrapper in enumerate(discriminator_wrappers):
            try:
                if wrapper.config.model_type == ModelType.OPENAI:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT_DETECTION},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT_DETECTION}\n\nUser: {prompt}"}]
                
                pred_answer = wrapper.generate(messages)
                print(f"Discriminator {i+1} answer: {pred_answer}")
                
                results.append("2" in pred_answer.lower())
            except Exception as e:
                print(f"Error with discriminator {i+1}: {e}")
                results.append(False)
        
        return results, hallucinated_answer, justification
    except Exception as e:
        print(f"Error in check_hallu_multiple: {e}")
        return [False] * len(discriminator_wrappers), None, None

def generate_hallucinations(
    question: str, 
    knowledge: str, 
    ground_truth: str, 
    generator_wrapper: LLMWrapper,
    discriminator_wrappers: List[LLMWrapper],
    sent_model: SentenceTransformer
) -> Tuple[List[Dict], Optional[str], DifficultyLevel]:
    """Generate and evaluate hallucinations with TextGrad-based improvement."""
    hallucinations = []
    current_system_prompt = SYSTEM_PROMPT
    best_difficulty = DifficultyLevel.EASY
    attempts_on_current_datapoint = 0
    
    while attempts_on_current_datapoint < NUM_GENERATIONS:
        try:
            print(f"\nAttempt {attempts_on_current_datapoint + 1}/{NUM_GENERATIONS} for current datapoint")
            
            # 1. Fresh generation for each attempt
            prompt = f"#Question#: {question}\n\n#Knowledge#: {knowledge}\n\n#Ground truth answer#: {ground_truth}\n\n#Hallucinated Answer#:"
            
            if generator_wrapper.config.model_type == ModelType.OPENAI:
                messages = [{"role": "system", "content": current_system_prompt},
                          {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": f"{current_system_prompt}\n\n{prompt}"}]
            
            generated_text = generator_wrapper.generate(messages, max_new_tokens=MAX_NEW_TOKENS)
            
            if not generated_text:
                print(f"No text generated in attempt {attempts_on_current_datapoint+1}")
                attempts_on_current_datapoint += 1
                continue
            
            current_text = generated_text
            if generator_wrapper.config.model_type == ModelType.HUGGINGFACE:
                long_answer = f"{prompt} {current_text}"
            else:
                long_answer = current_text
            
            # 2. Check this generation
            discriminator_results, hallucinated_answer, justification = check_hallu_multiple(
                long_answer, discriminator_wrappers
            )
            
            # If any discriminator is fooled, record and break immediately
            if any(discriminator_results):
                difficulty = determine_difficulty(discriminator_results)
                current_hallucination = {
                    'answer': hallucinated_answer if hallucinated_answer else current_text,
                    'justification': justification,
                    'discriminator_results': discriminator_results,
                    'difficulty': difficulty,
                    'attempt_number': attempts_on_current_datapoint + 1
                }
                hallucinations.append(current_hallucination)
                if difficulty.value > best_difficulty.value:
                    best_difficulty = difficulty
                break  # Exit immediately as we've fooled at least one discriminator
            
            # 3. If no discriminator was fooled, try TextGrad improvement
            print(f"Attempt failed. Using TextGrad to improve...")
            try:
                improved_text = improve_hallucination_with_textgrad(
                    long_answer, question, knowledge
                )
                
                # Generate based on TextGrad's improvements
                if generator_wrapper.config.model_type == ModelType.OPENAI:
                    messages = [
                        {"role": "system", "content": current_system_prompt},
                        {"role": "user", "content": f"{prompt}\n\nPrevious attempt improved by TextGrad: {improved_text}"}
                    ]
                else:
                    messages = [{"role": "user", "content": f"{current_system_prompt}\n\n{prompt}\n\nPrevious attempt improved by TextGrad: {improved_text}"}]
                
                improved_generated = generator_wrapper.generate(messages, max_new_tokens=MAX_NEW_TOKENS)
                improved_long_answer = improved_generated if generator_wrapper.config.model_type == ModelType.OPENAI else f"{prompt} {improved_generated}"
                
                # Check the improved generation
                discriminator_results, hallucinated_answer, justification = check_hallu_multiple(
                    improved_long_answer, discriminator_wrappers
                )
                
                # If TextGrad improvement fooled any discriminator, record and break
                if any(discriminator_results):
                    difficulty = determine_difficulty(discriminator_results)
                    current_hallucination = {
                        'answer': hallucinated_answer if hallucinated_answer else improved_generated,
                        'justification': justification,
                        'discriminator_results': discriminator_results,
                        'difficulty': difficulty,
                        'attempt_number': attempts_on_current_datapoint + 1
                    }
                    hallucinations.append(current_hallucination)
                    if difficulty.value > best_difficulty.value:
                        best_difficulty = difficulty
                    break  # Exit as TextGrad improvement fooled a discriminator
                    
            except Exception as e:
                print(f"Error in TextGrad optimization: {e}")
            
            # 4. Record failed attempt
            difficulty = determine_difficulty(discriminator_results)
            current_hallucination = {
                'answer': hallucinated_answer if hallucinated_answer else current_text,
                'justification': justification,
                'discriminator_results': discriminator_results,
                'difficulty': difficulty,
                'attempt_number': attempts_on_current_datapoint + 1
            }
            hallucinations.append(current_hallucination)
            
            # Move to next attempt only if we haven't fooled any discriminator
            attempts_on_current_datapoint += 1
                
        except Exception as e:
            print(f"Error in generation attempt {attempts_on_current_datapoint+1}: {e}")
            attempts_on_current_datapoint += 1
            continue
    
    # If we never fooled any discriminator, find least similar answer
    if not any(any(h['discriminator_results']) for h in hallucinations):
        try:
            similarities = [calculate_semantic_similarity(h['answer'], ground_truth, sent_model) 
                          for h in hallucinations if h['answer']]
            if similarities:
                max_index = similarities.index(max(similarities))
                final_answer = hallucinations[max_index]['answer']
            else:
                final_answer = None
        except Exception as e:
            print(f"Error calculating similarities: {e}")
            final_answer = None
    else:
        # Use the first successful hallucination
        successful = next(h for h in hallucinations if any(h['discriminator_results']))
        final_answer = successful['answer']
    
    return hallucinations, final_answer, best_difficulty
    
# ============ Main Function ============
def main():
    """Main execution function."""
    
    # Update global variables based on arguments
    global SYSTEM_PROMPT, SYSTEM_PROMPT_DETECTION, BATCH_SIZE, OUTPUT_FILE, CHECKPOINT_FILE
    SYSTEM_PROMPT = load_prompt("./Prompts/system_prompt_medical.txt")
    SYSTEM_PROMPT_DETECTION = load_prompt("./Prompts/system_prompt_detection.txt")
    BATCH_SIZE = 9000
    OUTPUT_FILE = os.path.join(" ", '') # Placeholder for output file path
    CHECKPOINT_FILE = os.path.join(" ", ' ') # Placeholder for checkpoint file path
    
    print("Initializing models...")
    generator_wrapper, discriminator_wrappers, sent_model = initialize_models(
        openai_api_key=" ",  # Placeholder for OpenAI API key
    )
    
    results_df = create_empty_results_df()
    
    print("Loading dataset...")
    df = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split=f"train[:{BATCH_SIZE}]")
    
    print(f"Processing {len(df['question'])} questions...")
    for i in tqdm(range(len(df['question']))):
        try:
            question = df['question'][i]
            knowledge = df['context'][i]
            ground_truth = df['long_answer'][i]
            
            print(f"\nProcessing question {i+1}/{len(df['question'])}...")
            hallucinations, least_similar_answer, final_difficulty = generate_hallucinations(
                question, knowledge, ground_truth, 
                generator_wrapper, discriminator_wrappers, sent_model
            )
            
            row_data = {
                'question': question,
                'knowledge': knowledge,
                'ground_truth': ground_truth,
                'least_similar_answer': least_similar_answer,
                'final_difficulty_level': final_difficulty.value
            }
            
            for j, hall in enumerate(hallucinations):
                if hall:
                    row_data[f'hallucinated_answer_{j+1}'] = hall['answer']
                    row_data[f'justification_{j+1}'] = hall['justification']
                    for k, result in enumerate(hall['discriminator_results']):
                        row_data[f'fooled_discriminator_{k+1}_{j+1}'] = result
                    row_data[f'difficulty_level_{j+1}'] = hall['difficulty'].value
                else:
                    row_data[f'hallucinated_answer_{j+1}'] = None
                    row_data[f'justification_{j+1}'] = None
                    for k in range(len(discriminator_wrappers)):
                        row_data[f'fooled_discriminator_{k+1}_{j+1}'] = False
                    row_data[f'difficulty_level_{j+1}'] = DifficultyLevel.EASY.value
            
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
            save_checkpoint(results_df)
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            continue
    
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Final results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()