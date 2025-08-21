print("Loading libraries...")
import os
import json
import html
import ast
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

ATOMIC_CSV_PATH = "./content/atomic_data.csv" 
MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"                
TEST_JSONL_FILE = "./content/train.csv.jsonl" 
OUTPUT_CSV = "results_output.csv"                

# Set this if you want to work offline with downloaded models
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# DATA LOADERS

def load_test_data(csv_path):
    """Load test data from CSV file"""
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        data.append({
            "id": row['id'],
            "statement": row['text'].strip(),
            "anger": row['anger'],
            "fear": row['fear'], 
            "joy": row['joy'],
            "sadness": row['sadness'],
            "surprise": row['surprise']
        })
    return data

def parse_list(cell):
    """Parse string representation of lists in ATOMIC dataset into actual lists"""
    if isinstance(cell, list):
        return cell
    try:
        return ast.literal_eval(cell) if isinstance(cell, str) else cell
    except Exception:
        return cell

def dedup_list(lst):
    """Remove duplicates from a list while preserving order"""
    if not isinstance(lst, list):
        return lst
    seen = set()
    filtered = []
    for item in lst:
        norm_item = item.strip().lower()
        if norm_item in {"none", ""}:
            continue
        if norm_item not in seen:
            seen.add(norm_item)
            filtered.append(item.strip())
    return filtered

def load_and_process_atomic(filepath):
    """Load and process ATOMIC dataset"""
    df_atomic = pd.read_csv(filepath)
    
    list_columns = ["oEffect", "oReact", "oWant", "xAttr", "xEffect",
                    "xIntent", "xNeed", "xReact", "xWant"]
    
    for col in list_columns:
        df_atomic[col] = df_atomic[col].apply(parse_list).apply(dedup_list)
    
    def create_combined_text(row):
        # Create a combined text representation for embedding
        parts = [
            f"Event: {row['event']}",
            f"Attribute of PersonX: {row['xAttr']}",
            f"Effect on PersonX: {row['xEffect']}",
            f"Intent of PersonX: {row['xIntent']}",
            f"Need of PersonX: {row['xNeed']}",
            f"Reaction of PersonX: {row['xReact']}",
            f"Want of PersonX: {row['xWant']}",
            f"Effect on others: {row['oEffect']}",
            f"Reaction of others: {row['oReact']}",
            f"Want of others: {row['oWant']}"
        ]
        return " | ".join(parts)
    
    df_atomic["combined"] = df_atomic.apply(create_combined_text, axis=1)
    return df_atomic

# FAISS & EMBEDDINGS

def build_faiss_index(embeddings):
    """Build a FAISS index for fast similarity search"""
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def retrieve_atomic_entries(query_text, k, embedder, index, df_atomic):
    """Retrieve the k most similar entries from ATOMIC"""
    query_embedding = embedder.encode([query_text],
                                    convert_to_tensor=False,
                                    encode_kwargs={"return_attention_mask": True}
                                    )[0].astype("float32")
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    
    # Modify columns to match the ones you want to retrieve
    return df_atomic.iloc[indices[0]][["event", "oEffect", "xEffect", "oReact", "xReact", "oWant", "xWant"]].to_dict("records")

def build_prompt(statement_text, context_knowledge=None):
    """
    Build a prompt for the LLM classification task
    Customize this function based on specific classification task
    """
    context_str = ""
    if context_knowledge:
        context_sections = []
        for i, entry in enumerate(context_knowledge, 1):
            context_parts = [
                f"### Related Situation {i}: '{entry['event']}'",
                f"• Effects on others: {', '.join(str(x) for x in entry['oEffect']) if isinstance(entry['oEffect'], list) else entry['oEffect']}",
                f"• Effects on person: {', '.join(str(x) for x in entry['xEffect']) if isinstance(entry['xEffect'], list) else entry['xEffect']}",
                f"• How others react: {', '.join(str(x) for x in entry['oReact']) if isinstance(entry['oReact'], list) else entry['oReact']}",
                f"• How person reacts: {', '.join(str(x) for x in entry['xReact']) if isinstance(entry['xReact'], list) else entry['xReact']}",
                f"• What others want: {', '.join(str(x) for x in entry['oWant']) if isinstance(entry['oWant'], list) else entry['oWant']}",
                f"• What person wants: {', '.join(str(x) for x in entry['xWant']) if isinstance(entry['xWant'], list) else entry['xWant']}"
            ]
            context_sections.append("\n".join(context_parts))
        
        context_str = "\n### Contextual Knowledge from Similar Situations:\n" + "\n\n".join(context_sections)
    
    prompt = f"""You are an expert in emotion analysis and psychological assessment.

### Task:
Analyze the emotional intensity in the given statement. For each emotion (anger, fear, joy, sadness, surprise), rate the intensity on a scale of 0-3:
- 0: No emotion present
- 1: Low intensity
- 2: Moderate intensity  
- 3: High intensity

### Statement:
{statement_text}

{context_str}

### Response Format:
Anger: [0-3]
Fear: [0-3]
Joy: [0-3]
Sadness: [0-3]
Surprise: [0-3]

### Response:
"""
    return prompt 
    
class ModelPredictor:
    def __init__(self, model_path, dtype=torch.float16):
        """Initialize the model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=dtype
        )
        print(f"Model loaded on device: {self.model.device}")

    def batch_generate(self, prompts, max_new_tokens=300): ## You can change the amount of max new tokens (this is basically how many new tokens will the model generate)
        """Generate responses for a batch of prompts"""
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

# BATCH PREDICTION

BATCH_SIZE = 16  # Adjust based on your GPU memory

def batched_predict(statements_data, k, embedder, index, df_atomic, predictor, batch_size=BATCH_SIZE, use_context=False):
    """Run predictions in batches"""
    results = []
    for i in tqdm(range(0, len(statements_data), batch_size), desc="Batch inference"):
        batch = statements_data[i:i + batch_size]
        prompts = []
        meta_data = []

        for rec in batch:
            statement = rec["statement"]

            context_knowledge = []
            if use_context:
                context_knowledge = retrieve_atomic_entries(statement, k, embedder, index, df_atomic)

            prompt = build_prompt(statement, context_knowledge) 
            prompts.append(prompt)

            meta_data.append({
                "id": rec["id"],
                "statement": statement,
                "true_anger": rec["anger"],
                "true_fear": rec["fear"],
                "true_joy": rec["joy"], 
                "true_sadness": rec["sadness"],
                "true_surprise": rec["surprise"]
})

        outputs = predictor.batch_generate(prompts)
        
        for output, meta in zip(outputs, meta_data):
            results.append({
                "statement": meta["statement"],
                "model_output": output,
                "true_label": meta["true_label"]
            })

    return results

# MAIN

if __name__ == "__main__":
    print("Loading test data...")
    test_data = load_test_data(TEST_JSONL_FILE)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    
    print("Loading and processing ATOMIC dataset...")
    df_atomic = load_and_process_atomic(ATOMIC_CSV_PATH)
    texts = df_atomic["combined"].tolist()

    print("Embedding ATOMIC entries...")
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  ## You can also use a different embedding model 
    embeddings = np.array(embedder.encode(
                                        texts,
                                        convert_to_tensor=False,
                                        encode_kwargs={"return_attention_mask": True}
                                        )).astype("float32")
    index = build_faiss_index(embeddings)

    print("Loading LLM model...")
    predictor = ModelPredictor(model_path=MODEL_PATH)

    print("Running predictions...")
    results = batched_predict(
        statements_data=test_data,
        k=3,  # Number of similar entries to retrieve from ATOMIC dataset
        embedder=embedder,
        index=index,
        df_atomic=df_atomic,
        predictor=predictor,
        batch_size=BATCH_SIZE,
        use_context=True  # Set to False if you don't want to use RAG context
    )
    print("Predictions completed.")

    print("Saving results...")
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Results saved to {OUTPUT_CSV}")
    
  