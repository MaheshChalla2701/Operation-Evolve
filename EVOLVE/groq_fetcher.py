import os
import json
import torch
import random
import requests
import logging
import tiktoken
from config import EvolveConfig
from data import SyntheticDataset, TextDataset

logger = logging.getLogger("evolve.groq_fetcher")

TOPICS = [
    "Current Tech News",
    "Global Weather Patterns",
    "Stock Market Trends",
    "Sports Analytics",
    "Quantum Computing",
    "Space Exploration",
    "Climate Change Indicators",
    "Social Media Sentiment",
    "Coding",
    "general knowledge",
    "history",
    "geography",
    "science",
    "math",
    "art",
    "music",
    "sports",
    "movies",
    "books",
    "technology",
    "business",
    "health",
    "fitness",
    "food",
    "travel",
    "fashion",
    "gaming",
    "politics",
    "economics",
    "psychology",
    "sociology",
    "philosophy",
    "religion",
    "mythology",
    "literature",
    "poetry",
    "drama",
    "fiction",
    "non-fiction",
    "biography",
    "autobiography",
    "memoir"
]

def _fetch_focused_topics(config: EvolveConfig, focus_prompt: str) -> list:
    """
    Ask Groq to generate domain-specific sub-topics for the given focus prompt.
    Returns a list of topic strings, falling back to random TOPICS on failure.
    """
    headers = {
        "Authorization": f"Bearer {config.groq_api_key}",
        "Content-Type": "application/json"
    }
    subtopic_prompt = f"""The user wants to train a machine learning model focused on: "{focus_prompt}".
Generate exactly {config.num_classes} distinct sub-topics within this domain that would form
separate, distinct data clusters useful for training.
Output ONLY a raw JSON list of {config.num_classes} short topic strings. No markdown, no explanation.
Example format: ["Topic A", "Topic B", "Topic C", "Topic D"]"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": config.llm_model_name,
                "messages": [{"role": "user", "content": subtopic_prompt}],
                "temperature": 0.7
            },
            timeout=20
        )
        response.raise_for_status()
        reply = response.json()['choices'][0]['message']['content'].strip()
        # Strip markdown if present
        if reply.startswith("```"):
            reply = reply.split("```")[1].strip()
            if reply.startswith("json"):
                reply = reply[4:].strip()
        topics = json.loads(reply)
        if isinstance(topics, list) and len(topics) == config.num_classes:
            logger.info(f"[Fetcher] 🎯 Focused sub-topics for '{focus_prompt}': {topics}")
            return topics
    except Exception as e:
        logger.warning(f"[Fetcher] Failed to get focused topics: {e}. Falling back to random topics.")
    return random.sample(TOPICS, config.num_classes)


def fetch_internet_dataset(config: EvolveConfig, focus_prompt: str = "") -> SyntheticDataset:
    """
    Fetches a dataset by asking Groq to generate 16-dimensional numeric cluster
    centers for topics. When focus_prompt is set, topics are domain-specific;
    otherwise random internet topics are used (original behaviour).
    """
    if config.loss_mode == "lm":
        return _fetch_text_dataset(config, focus_prompt)

    if not config.groq_api_key:
        logger.warning("No GROQ_API_KEY found! Generating fallback synthetic dataset.")
        return _generate_fallback(config)

    # Pick topics — focused or generic
    if focus_prompt:
        chosen_topics = _fetch_focused_topics(config, focus_prompt)
    else:
        chosen_topics = random.sample(TOPICS, config.num_classes)
    
    prompt = f"""You are a data generation system. I need you to invent 16-dimensional numeric representations (vectors) for the following {config.num_classes} topics:
{', '.join(chosen_topics)}

Output ONLY a raw JSON object (do not use Markdown blocks like ```json). 
The format MUST be exactly:
{{
  "topic_0_center": [float, float, ... 16 floats],
  "topic_1_center": [float, float, ... 16 floats],
  "topic_2_center": [float, float, ... 16 floats],
  "topic_3_center": [float, float, ... 16 floats]
}}
Ensure the floats are between -5.0 and +5.0 and distinctly different across topics so they form separate clusters.
"""

    headers = {
        "Authorization": f"Bearer {config.groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": config.llm_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        reply = response.json()['choices'][0]['message']['content'].strip()
        
        # Strip markdown if the AI hallucinated it
        if reply.startswith("```json"):
            reply = reply[7:-3].strip()
        if reply.startswith("```"):
            reply = reply[3:-3].strip()
            
        centers_data = json.loads(reply)
        
        # Verify formats map to expectations
        centers = []
        for i in range(config.num_classes):
            key = f"topic_{i}_center"
            vec = centers_data[key]
            if len(vec) != config.input_dim:
                # pad or slice
                if len(vec) > config.input_dim: vec = vec[:config.input_dim]
                else: vec = vec + [0.0]*(config.input_dim - len(vec))
            centers.append(vec)
            
        centers_tensor = torch.tensor(centers, dtype=torch.float32)
        logger.info(f"Successfully fetched new dataset cluster centers from internet topics: {chosen_topics}")

    except Exception as e:
        logger.error(f"Failed to fetch data from Groq API: {e}. Using fallback.")
        return _generate_fallback(config)

    # Now populate actual samples around these centers with noise
    samples_per_class = max(1, config.groq_dataset_size // config.num_classes)
    
    all_feats = []
    all_labels = []
    
    for cls in range(config.num_classes):
        center = centers_tensor[cls]
        # create noisy points
        noise = torch.randn(samples_per_class, config.input_dim) * 0.2
        feats = center.unsqueeze(0).expand(samples_per_class, -1) + noise
        labels = torch.full((samples_per_class,), cls, dtype=torch.long)
        
        all_feats.append(feats)
        all_labels.append(labels)
        
    features = torch.cat(all_feats)
    labels = torch.cat(all_labels)
    
    # Shuffle dataset
    perm = torch.randperm(features.shape[0])
    return SyntheticDataset(features[perm], labels[perm])

def _generate_fallback(config: EvolveConfig) -> SyntheticDataset:
    from data import generate_seed_data
    logger.info("Generating standard synthetic dataset as fallback.")
    return generate_seed_data(
        n=config.groq_dataset_size,
        num_classes=config.num_classes,
        input_dim=config.input_dim,
        noise_std=0.2
    )

def _fetch_text_dataset(config: EvolveConfig, focus_prompt: str) -> TextDataset:
    if not config.groq_api_key:
        logger.warning("No GROQ_API_KEY found! Generating fallback synthetic text dataset.")
        from data import generate_text_seed_data
        return generate_text_seed_data(config.groq_dataset_size, config.max_seq_len, config.vocab_size)

    if focus_prompt:
        chosen_topics = _fetch_focused_topics(config, focus_prompt)
    else:
        chosen_topics = random.sample(TOPICS, config.num_classes)
        
    prompt = f"""Write a comprehensive, highly detailed article that covers the following topics:
{', '.join(chosen_topics)}

The article must be at least 1500 words long and contain complex reasoning and facts.
Output ONLY the text of the article. Do not include titles, markdown formatting, or introductory remarks."""

    headers = {
        "Authorization": f"Bearer {config.groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": config.llm_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        text = response.json()['choices'][0]['message']['content'].strip()
        
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        
        seqs = []
        for i in range(0, len(tokens) - config.max_seq_len - 1, config.max_seq_len):
            seqs.append(tokens[i : i + config.max_seq_len])
            if len(seqs) >= config.groq_dataset_size:
                break
                
        # Pad data if necessary
        while len(seqs) > 0 and len(seqs) < config.groq_dataset_size:
            seqs.append(seqs[len(seqs) % len(seqs)])
            
        if len(seqs) == 0:
             raise ValueError("Generated text was too short to form any sequence.")

        features = torch.tensor(seqs, dtype=torch.long)
        logger.info(f"Successfully fetched text dataset from topics: {chosen_topics} ({len(seqs)} sequences)")
        return TextDataset(features)

    except Exception as e:
        logger.error(f"Failed to fetch text data from Groq: {e}. Using fallback.")
        from data import generate_text_seed_data
        return generate_text_seed_data(config.groq_dataset_size, config.max_seq_len, config.vocab_size)

if __name__ == "__main__":
    # Test execution
    from utils import setup_logging
    setup_logging("INFO")
    cfg = EvolveConfig()
    dataset = fetch_internet_dataset(cfg)
    print(f"Dataset Shape: {dataset.features.shape}")
