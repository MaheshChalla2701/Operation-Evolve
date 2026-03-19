import os
import copy
import json
import requests
from typing import Any, Dict, List, Tuple

# Groq API Key injected by default, or loaded from environment
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "[ENCRYPTION_KEY]")

class EvolutionAgent:
    """
    LLM-Powered Evolution Agent using Groq (Llama 3).
    Analyzes model metrics and proposes configuration changes via a prompt.
    """
    def __init__(self):
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = GROQ_API_KEY

    def analyze(self, config: dict, metrics: dict) -> tuple[dict, list]:  # type: ignore
        """
        Analyze metrics and produce a mutated configuration via Groq API.
        Returns: (proposed_config, list_of_mutation_strings)
        """
        if not self.api_key:
            print("  [AGENT] ⚠️ No GROQ_API_KEY found. Skipping mutation.")
            return copy.deepcopy(config), []

        # 1. Build the prompt context
        history = metrics.get("history", [])
        if history is None:
            history = []
            
        load_dist = metrics.get("load_distribution", [])
        if load_dist is None:
            load_dist = []
            
        system_prompt = """You are the 'Evolution Agent', a highly intelligent AI guiding the architecture search of a sparse Mixture-of-Experts (MoE) Transformer.
Your job is to read the model's current configuration and performance, then propose exactly ONE upgraded `config.json` configuration that will improve learning.

THE RULES:
1. Propose intelligent structural changes (e.g., adding/removing `num_experts`, changing `expert_hidden_dim`) OR hyperparameter tweaks (`learning_rate`, `router_temperature`, `top_k`).
2. The evaluation pipeline supports Speculative Acceptance: if you propose a structural change, the model is guaranteed to run a full training cycle to see if the new structure learns, before deciding to revert. So you can safely explore adding experts!
3. Be conservative and deliberate. Do not blow up the model size instantly.
4. If an expert utilization load is 0.0 or near 0, you might consider pruning experts or increasing router_temperature.
5. If one expert dominates 80%+ of the tokens, consider cloning it by adding an expert or decreasing router_temperature.
6. YOU MUST RETURN EXACTLY VALID JSON MATCHING THIS SCHEMA:
{
  "proposed_config": {
    "num_experts": int,
    "top_k": int,
    "expert_hidden_dim": int,
    "learning_rate": float,
    "router_temperature": float
  },
  "mutations": ["String explaining what you changed and exactly why"]
}
If you propose NO changes because the model is perfect, return the original config and an empty [] mutations list. Output ONLY JSON.
"""

        user_prompt = f"""
CURRENT CONFIGURATION:
{json.dumps(config, indent=2)}

RECENT PERFORMANCE METRICS:
Final Train Loss: {metrics.get('final_train_loss')}
Best Val Loss: {metrics.get('best_val_loss')}
Accuracy: {metrics.get('accuracy', 0)*100:.2f}%
Expert Load Distribution (Utilization % per expert): {load_dist}

HISTORY (Last few validation checkpoints):
"""
        for h in history[-4:]:
             user_prompt += f"  Step {h.get('step')}: Train Loss {h.get('train_loss')}, Val Loss {h.get('val_loss')}, Acc {h.get('accuracy')}\n"

        user_prompt += "\nPlease propose the next optimal configuration. Output ONLY valid JSON containing the new config."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Use Llama 3 70B for fast reasoning via Groq
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"}
        }

        try:
            print("  [AGENT] 🧠 Asking Groq (Llama 3) for architectural advice...")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            llm_content = data['choices'][0]['message']['content'].strip()
            
            # 2. Parse JSON response (cleans out markdown wrappers if Grok adds them)
            if "```json" in llm_content:
                llm_content = llm_content.split("```json")[-1].split("```")[0].strip()
            elif "```" in llm_content:
                llm_content = llm_content.split("```")[-1].split("```")[0].strip()

            result = json.loads(llm_content)
            proposed_config = result.get("proposed_config", copy.deepcopy(config))
            mutations = result.get("mutations", [])
            
            # 3. Safety Fallback: Ensure config wasn't destroyed
            if "num_experts" not in proposed_config or "learning_rate" not in proposed_config:
                print("  [AGENT] ⚠️ Grok returned invalid config schema. Preserving original.")
                return copy.deepcopy(config), []
                
            # Copy over hidden/locked vars using the original config if Grok forgot them
            for key in config.keys():
                if key not in proposed_config:
                    proposed_config[key] = config[key]

            # Enforce bounds roughly
            proposed_config["num_experts"] = max(2, min(8, int(proposed_config.get("num_experts", config["num_experts"]))))
            proposed_config["top_k"] = max(1, min(proposed_config["num_experts"], int(proposed_config.get("top_k", config.get("top_k", 1)))))
            
            return proposed_config, mutations

        except json.JSONDecodeError as e:
            print(f"  [AGENT] ⚠️ Grok returned invalid JSON: {e}")
            return copy.deepcopy(config), []
        except Exception as e:
            print(f"  [AGENT] ⚠️ Grok API Call Failed: {e}")
            return copy.deepcopy(config), []
