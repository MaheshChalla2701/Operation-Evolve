import os
import json
import torch
import random
import requests
import logging
from config import EvolveConfig
from data import SyntheticDataset

logger = logging.getLogger("evolve.groq_fetcher")

TOPICS = [
    "Current Tech News",
    "Global Weather Patterns",
    "Stock Market Trends",
    "Sports Analytics",
    "Quantum Computing",
    "Space Exploration",
    "Climate Change Indicators",
    "Social Media Sentiment"
]

def fetch_internet_dataset(config: EvolveConfig) -> SyntheticDataset:
    """
    Simulates fetching distinct "internet datasets" by asking Groq to generate
    16-dimensional numeric cluster centers for 4 random internet topics.
    Then, it generates a full dataset evenly distributed around those cluster centers.
    """
    if not config.groq_api_key:
        logger.warning("No GROQ_API_KEY found! Generating fallback synthetic dataset.")
        return _generate_fallback(config)

    # Pick 4 random topics for this iteration
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

if __name__ == "__main__":
    # Test execution
    from utils import setup_logging
    setup_logging("INFO")
    cfg = EvolveConfig()
    dataset = fetch_internet_dataset(cfg)
    print(f"Dataset Shape: {dataset.features.shape}")
