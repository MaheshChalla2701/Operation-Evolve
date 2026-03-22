import os
import copy
import json
import requests
from typing import Any, Dict, List, Tuple

# Groq API Key injected by default, or loaded from environment
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Set via: export GROQ_API_KEY=your_key_here

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
Your job is to read the model's current configuration and performance metrics, then propose EITHER:
  A) Configuration hyperparameter changes, OR
  B) A Python code patch (a short snippet) that modifies model.py or trainer.py to improve performance

THE RULES:
1. Prefer config changes for small adjustments (learning_rate, num_experts, top_k, router_temperature, expert_hidden_dim).
2. Propose a code_patch if you want to add a new technique (e.g., different activation function, gradient clipping change, dropout schedule).
   - A code_patch is a Python string that patches the running config dict or adds a note for the human to apply. Keep it SHORT and safe.
3. Be conservative. Do not blow up model size instantly.
4. If expert load is 0.0 or near 0 for an expert → prune experts or raise router_temperature.
5. If one expert handles 80%+ of tokens → clone it (add expert) or lower router_temperature.
6. YOU MUST RETURN EXACTLY VALID JSON MATCHING THIS SCHEMA:
{
  "proposed_config": {
    "num_experts": int,
    "top_k": int,
    "expert_hidden_dim": int,
    "learning_rate": float,
    "router_temperature": float
  },
  "mutations": ["String explaining what you changed and exactly why"],
  "code_patch": "Optional short Python snippet describing a code-level change to try. Empty string if none."
}
Output ONLY JSON.
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

        user_prompt += "\nPlease propose the next optimal configuration or code change. Output ONLY valid JSON."

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
            code_patch = result.get("code_patch", "")

            # Log code_patch if the agent proposed one
            if code_patch and code_patch.strip():
                print(f"  [AGENT] 💡 Code patch proposed: {code_patch[:200]}..." if len(code_patch) > 200 else f"  [AGENT] 💡 Code patch proposed:\n{code_patch}")
                mutations.append(f"[CODE PATCH] {code_patch[:150]}")
            
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
