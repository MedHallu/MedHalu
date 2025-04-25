#!/usr/bin/env python3
import torch
import gc
import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
# vLLM imports
from vllm import LLM, SamplingParams
import torch
import multiprocessing
import ast

# (If using the OpenAI branch, uncomment and set your API key)
# import openai
# openai.api_key = "YOUR_OPENAI_API_KEY"

# ---------------------
# PROMPT CREATION
# ---------------------
system_prompt = """
You are an AI assistant with extensive knowledge in the field of medicine. I want you act as an answer judge. Given a question and an answer, your objective is to determine if the
provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based
on the following hallucination types and the world knowledge.

Type: Misinterpretation of #Question#: These are hallucinated answers that misunderstands the question, leading to an off-topic or irrelevant response.
Type: Incomplete Information: These are hallucinated answers that Point out what is not true without providing correct information.
Type: Mechanism and Pathway Misattribution - These are hallucinated answer that falsely attribution of biological mechanisms, molecular pathways, or disease processes that contradicts established medical knowledge
Type: Methodological and Evidence Fabrication - Inventing false research methods, statistical data, or specific clinical outcomes


You are trying to determine if there is a factual contradiction between the answer and the world knowledge.
Some information in the answer might be fabricated. Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the factuality and want to pass. If you are unsure about the answer choose `2` instead of guessing and making a mistake. Do not return anything else, just the answer.
"""

def create_prompt(question, option1):
    prompt = f"""
Question: {question}
Answer: {option1}

Reutrn just the answer, Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the answer and want to pass. Dont return anything else, just the answer. Dont be verbose
If you are unsure about the answer choose `2` instead of guessing and making a mistake.
Your Judgement:
"""
    return prompt

def create_prompt_withknowledge(question, option1, knowledge):
    prompt = f"""
World Knowledge: {knowledge}
Question: {question}
Answer: {option1}

Reutrn just the answer, Answer '0' if the answer is factual and '1' if the answer is hallucinated and `2` if you are unsure about the answer and want to pass. Dont return anything else, just the answer. Dont be verbose
If you are unsure about the answer choose `2` instead of guessing and making a mistake.
Your Judgement:
"""
    return prompt

# ---------------------
# GPU MEMORY CLEARING
# ---------------------
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

# ---------------------
# METRICS CALCULATION
# ---------------------
def calculate_metrics(answer_list, llm_answers, df, model_config, use_knowledge):
    # Parse llm_answers into integers (0, 1, 2)
    llm_answers_int = []
    for i in llm_answers:
        i_lower = i.lower()
        if any(x in i_lower for x in ['1', 'not', 'non']):
            llm_answers_int.append(1)
        elif any(x in i_lower for x in ['not sure', 'pass', 'skip', '2']):
            llm_answers_int.append(2)
        else:
            llm_answers_int.append(0)

    answer_int = [int(i) for i in answer_list]

    df['llm_answers_int'] = llm_answers_int
    df['answer_int'] = answer_int
    df['Decision'] = [
        'Correct' if llm_answers_int[i] == answer_int[i]
        else 'Not Sure' if llm_answers_int[i] == 2
        else 'Incorrect'
        for i in range(len(llm_answers_int))
    ]

    # Difficulty-level metrics (filter out "Not Sure" predictions for metric calculation)
    difficulty_indices = {
        'easy':   [i for i, diff in enumerate(df['final_difficulty_level']) if diff == 'easy'],
        'medium': [i for i, diff in enumerate(df['final_difficulty_level']) if diff == 'medium'],
        'hard':   [i for i, diff in enumerate(df['final_difficulty_level']) if diff == 'hard']
    }

    metrics = {}
    for difficulty in ['easy', 'medium', 'hard']:
        indices = difficulty_indices[difficulty]
        if not indices:
            metrics[f'{difficulty}_accuracy']  = None
            metrics[f'{difficulty}_precision'] = None
            metrics[f'{difficulty}_recall']    = None
            metrics[f'{difficulty}_f1']        = None
            metrics[f'{difficulty}_percent_of_time_not_sure_chosen'] = None
            continue

        diff_answers = [answer_int[i] for i in indices]
        diff_llm = [llm_answers_int[i] for i in indices]

        # Calculate percentage of "Not Sure" responses
        not_sure_count = sum(1 for ans in diff_llm if ans == 2)
        metrics[f'{difficulty}_percent_of_time_not_sure_chosen'] = (not_sure_count / len(diff_llm))

        # Filter out cases where the model was not sure (i.e., where prediction is 2)
        valid_idx = [j for j, pred in enumerate(diff_llm) if pred != 2]
        if valid_idx:
            filtered_true = [diff_answers[j] for j in valid_idx]
            filtered_pred = [diff_llm[j] for j in valid_idx]
            metrics[f'{difficulty}_accuracy']  = accuracy_score(filtered_true, filtered_pred)
            metrics[f'{difficulty}_precision'] = precision_score(filtered_true, filtered_pred, zero_division=0)
            metrics[f'{difficulty}_recall']    = recall_score(filtered_true, filtered_pred, zero_division=0)
            metrics[f'{difficulty}_f1']        = f1_score(filtered_true, filtered_pred, zero_division=0)
        else:
            metrics[f'{difficulty}_accuracy']  = None
            metrics[f'{difficulty}_precision'] = None
            metrics[f'{difficulty}_recall']    = None
            metrics[f'{difficulty}_f1']        = None

    # Overall metrics (filtering out "Not Sure" predictions)
    valid_indices = [i for i, ans in enumerate(llm_answers_int) if ans != 2]
    if valid_indices:
        filtered_answers = [answer_int[i] for i in valid_indices]
        filtered_llm = [llm_answers_int[i] for i in valid_indices]
        metrics.update({
            'Model Name': model_config['model_name'],
            'Knowledge': 'Yes' if use_knowledge else 'No',
            'precision': precision_score(filtered_answers, filtered_llm, zero_division=0),
            'recall':    recall_score(filtered_answers, filtered_llm, zero_division=0),
            'f1':        f1_score(filtered_answers, filtered_llm, zero_division=0)
        })
    else:
        metrics.update({
            'Model Name': model_config['model_name'],
            'Knowledge': 'Yes' if use_knowledge else 'No',
            'precision': 0,
            'recall': 0,
            'f1': 0
        })

    not_sure_total = sum(1 for ans in llm_answers_int if ans == 2)
    metrics['overall_percent_of_time_not_sure_chosen'] = (not_sure_total / len(llm_answers_int)) if llm_answers_int else None

    return pd.DataFrame([metrics])

# ---------------------
# EVALUATION FUNCTION
# ---------------------
def run_evaluation(model_config, df, use_knowledge=False):
    chosen_answer_indices = []
    prompts = []
    answer_list = []   # ground-truth 0 or 1 for which answer is chosen
    
    for i in range(len(df)):
        question = df.loc[i, 'question']
        ground_truth = df.loc[i, 'ground_truth']
        hallucinated_answer = df.loc[i, 'least_similar_answer']
        
        if use_knowledge:
            # Assuming the 'knowledge' column is stored as a string representation of a dict.
            try:
                knowledge = ast.literal_eval(df.loc[i, 'knowledge'])['contexts']
            except Exception as e:
                knowledge = ""
        else:
            knowledge = None

        answers = [ground_truth, hallucinated_answer]
        random_val = random.randint(0, 1)
        chosen = answers[random_val]
        answer_list.append(random_val)
        
        if use_knowledge:
            user_prompt = create_prompt_withknowledge(question, chosen, knowledge)
        else:
            user_prompt = create_prompt(question, chosen)
        
        prompt_chat = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{system_prompt} {user_prompt}"},
        ]
        prompts.append(prompt_chat)

    llm_answers = []
    
    if model_config['type'] == 'hf':
        # Initialize vLLM model
        llm = LLM(
            model=model_config['model_name'],
            tensor_parallel_size=4,  # adjust based on your GPU setup
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            dtype=torch.float16,
        )
        tokenizer = llm.get_tokenizer()
        
        # Determine stop token IDs
        stop_tok_id = []
        if tokenizer.eos_token_id is not None:
            stop_tok_id.append(tokenizer.eos_token_id)
        for special_token in ["<|eot_id|>", "<|eos_token|>", "<end_of_turn>", "</s>"]:
            try:
                sid = tokenizer.convert_tokens_to_ids(special_token)
                if isinstance(sid, int) and sid not in stop_tok_id:
                    stop_tok_id.append(sid)
            except Exception:
                pass

        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.95,
            max_tokens=512,
            stop_token_ids=stop_tok_id
        )

        # Format prompts for batch generation
        batch_formatted_prompts = []
        for chat_prompt in prompts:
            formatted_prompt = tokenizer.apply_chat_template(
                chat_prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
            batch_formatted_prompts.append(formatted_prompt)
        
        outputs = llm.generate(batch_formatted_prompts, sampling_params)
        for out in outputs:
            text = out.outputs[0].text.strip()
            llm_answers.append(text)
            
        # Explicitly delete the model objects to free GPU memory
        del llm, tokenizer
        
    else:
        # Example for OpenAI calls (if needed)
        import openai
        for chat_prompt in prompts:
            response = openai.ChatCompletion.create(
                model=model_config['model_name'],
                messages=chat_prompt,
                max_tokens=4,
                n=1,
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            llm_answers.append(content)
    
    result_df = calculate_metrics(answer_list, llm_answers, df, model_config, use_knowledge)
    return result_df

# ---------------------
# FUNCTION TO RUN ONE EVALUATION IN A SUBPROCESS
# ---------------------
def evaluate_model_subprocess(model_config, use_knowledge, df_path, csv_path):
    try:
        # Each subprocess loads its own copy of the data
        df = pd.read_csv(df_path)
        print(f"Running {model_config['model_name']} with knowledge = {use_knowledge}")
        result = run_evaluation(model_config, df, use_knowledge)
        # Append results to CSV (create file with header if it does not exist)
        if os.path.exists(csv_path):
            result.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            result.to_csv(csv_path, mode='w', header=True, index=False)
    except Exception as e:
        print(f"Error evaluating {model_config['model_name']} with knowledge={use_knowledge}: {e}")
    finally:
        clear_gpu_memory()

# ---------------------
# MAIN FUNCTION
# ---------------------
def main():
    # Update these paths as needed
    df_path = " "
    csv_path = " "

    models = [
        {'type': 'hf', 'model_name': 'm42-health/Llama3-Med42-8B'},
        {'type': 'hf', 'model_name': 'OpenMeditron/Meditron3-8B'},
        {'type': 'hf', 'model_name': 'aaditya/OpenBioLLM-Llama3-8B'},
        {'type': 'hf', 'model_name': 'BioMistral/BioMistral-7B'},
        {'type': 'hf', 'model_name': 'TsinghuaC3I/Llama-3.1-8B-UltraMedical'},
        {'type': 'hf', 'model_name': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'},
        {'type': 'hf', 'model_name': 'Qwen/Qwen2.5-14B-Instruct'},
        {'type': 'hf', 'model_name': 'google/gemma-2-2b-it'},
        {'type': 'hf', 'model_name': 'google/gemma-2-9b-it'},
        {'type': 'hf', 'model_name': 'meta-llama/Llama-3.1-8B-Instruct'},
        {'type': 'hf', 'model_name': 'meta-llama/Llama-3.2-3B-Instruct'},
        {'type': 'hf', 'model_name': 'Qwen/Qwen2.5-7B-Instruct'},
        {'type': 'hf', 'model_name': 'Qwen/Qwen2.5-3B-Instruct'},
        # {'type': 'openai', 'model_name': 'gpt-4o-mini'}
    ]

    # For each model, run evaluation without knowledge and with knowledge sequentially in separate processes.
    for model_config in models:
        for use_knowledge in [False, True]:
            proc = multiprocessing.Process(
                target=evaluate_model_subprocess,
                args=(model_config, use_knowledge, df_path, csv_path)
            )
            proc.start()
            proc.join()  # Wait for the subprocess to finish before moving on
            print(f"Completed {model_config['model_name']} with knowledge = {use_knowledge}\n")

    print(f"All results saved to {csv_path}")

if __name__ == "__main__":
    main()
