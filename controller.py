import os
import shutil
import json
import time
import subprocess
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONFIG_FILE = "config.json"
HISTORY_FILE = "evolution_log.json"

# Set parameters for the Agentic LLM (Llama 3 over local, or externally hosted model)
# You mentioned Llama 3 / Groq / OpenAI in past sessions; here's a generic API format
# that works for OpenAI/Groq/Ollama APIs out of the box.
LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Default to Groq API
LLM_MODEL = "llama-3.3-70b-versatile"


def call_agent_for_new_config(current_config, current_metrics, history):
    """Prompts the LLM to analyze the config and propose A NEW CONFIG."""
    
    prompt = f"""You are the 'Operation Evolve' AI Controller. 
Your goal is to improve the Hierarchical Mixture-of-Experts (MoE) configuration by exploring different dimensions of hyperparameters and discovering structural optimizations.

CURRENT CONFIG: {json.dumps(current_config, indent=2)}
CURRENT METRICS REPORT CARD: {json.dumps(current_metrics, indent=2)}

RECENT EXPERIMENTS HISTORY:
{history} 
    
Propose a NEW configuration JSON. Choose carefully. You can tweak:
- n_embd (e.g., 64, 128) [previously d_model]
- expert_hidden_dim (e.g., 128, 256)
- num_groups (e.g., 2, 4, 8)
- experts_per_group (e.g., 2, 4, 8)
- learning_rate (e.g., 0.005, 0.001, 0.0005)
- num_heads (e.g., 2, 4, 8)
- num_layers (e.g., 1, 2, 4)  [Increases model depth]
- dropout (e.g., 0.0, 0.1, 0.2)
- batch_size (e.g., 4, 8, 16)
- router_temperature (e.g., 1.0, 1.5, 0.5) [Controls gating sharpness]
- vocab_size (Fix to 50257 for GPT-2 BPE)

OUTPUT DIRECTLY AS A PLAIN JSON OBJECT WITHOUT MARKDOWN OR CODE BLOCKS!
Your response must be a single JSON object containing two keys: "rationale" (a short explanation of why you made these specific changes based on the metrics) and "config" (the actual proposed configuration overrides).
Example:
{{
  "rationale": "I am increasing expert_hidden_dim to provide greater parameter capacity per expert, while reducing the learning_rate to ensure stability given the larger model size.",
  "config": {{"n_embd": 128, "expert_hidden_dim": 256, "num_groups": 4, "experts_per_group": 8, "learning_rate": 0.002, "num_heads": 4, "num_layers": 2, "dropout": 0.1, "batch_size": 8, "router_temperature": 1.0, "vocab_size": 50257}}
}}
"""
    print("🤖 Agent is analyzing the current state and generating a hypothesis...")
    
    # Send request to local Ollama (Llama3) - modify for Groq/OpenAI as needed
    try:
        if "localhost" in LLM_API_URL:
            # Ollama API Format
            response = requests.post(LLM_API_URL, json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }).json()
            response_text = response['message']['content']
        else:
            # OpenAI / Groq Format (Requires API Key in headers)
            headers = {
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            }
            response = requests.post(LLM_API_URL, headers=headers, json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }).json()
            
            # Check if there are errors from the API side
            if 'choices' not in response:
                print("❌ Groq API Error Response:", response)
                return None
                
            response_text = response['choices'][0]['message']['content']

        # Parse JSON from response
        # Crude extraction of JSON to handle sometimes chatty models
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        new_config_str = response_text[json_start:json_end]
        
        response_json = json.loads(new_config_str)
        
        rationale = response_json.get("rationale", "No rationale provided.")
        proposed_config = response_json.get("config", response_json) # Fallback if agent ignores instructions
        
        # Merge over current config to prevent missing keys
        merged_config = current_config.copy()
        merged_config.update(proposed_config)
        return merged_config, rationale
    except Exception as e:
        print(f"⚠️ Failed to get valid proposed config from AI: {e}")
        print(f"Raw Output: {response_text if 'response_text' in locals() else 'None'}")
        return None

def run_train_eval(config_to_test):
    """Executes the train_eval.py child process and extracts the metrics."""
    # Write the temporary config
    with open("proposed_config.json", "w") as f:
        json.dump(config_to_test, f)
        
    if os.path.exists("metrics.json"):
        os.remove("metrics.json")
        
    print("🔬 Executing objective model evaluation (train_eval.py)...")
    try:
        subprocess.run(
            ["python", "train_eval.py", "proposed_config.json"], 
            capture_output=True, text=True, check=True
        )
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                metrics = json.load(f)
            return metrics
        print("❌ metrics.json not found after execution.")
    except subprocess.CalledProcessError as e:
        print("❌ Model execution crashed during training.")
        print("Stderr:", e.stderr)
        
    return {"best_val_loss": float('inf')}  # Failed run means worst possible loss


def run_operation_evolve():
    print("🚀 Starting Operation Evolve (MoE Cycle)...\n")
    
    if not os.path.exists(CONFIG_FILE):
        print(f"Missing {CONFIG_FILE}. Run setup first.")
        return
        
    with open(CONFIG_FILE, "r") as f:
        current_config = json.load(f)
        
    # Read history buffer
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Could not parse existing {HISTORY_FILE}. Starting fresh.")
            history = []
            
    history_context = json.dumps(history[-10:], indent=2) # Pass last 10 cycles as context array
    # Baseline test
    baseline_metrics = run_train_eval(current_config)
    best_loss = baseline_metrics.get("best_val_loss", float('inf'))
    print(f"🎯 BASELINE LOSS: {best_loss:.6f}")
    
    # Evolution Loop (Single Cycle for demonstration. Change to `while True` for infinite)
    for cycle in range(1, 4):
        print(f"\n--- CYCLE {cycle} ---")
        
        # 1. Proposal
        proposal_result = call_agent_for_new_config(current_config, baseline_metrics, history_context)
        if not proposal_result:
            time.sleep(2)
            continue
            
        proposed_config, rationale = proposal_result
            
        print(f"🧠 Rationale: {rationale}")
        print("💡 Proposed changes:")
        for k, v in proposed_config.items():
            old_v = current_config.get(k)
            if old_v != v:
                print(f"   - {k}: {old_v} ➡️  {v}")
        
        # 2. Testing
        test_metrics = run_train_eval(proposed_config)
        test_loss = test_metrics.get("best_val_loss", float('inf'))
        print(f"   Score: {test_loss:.6f}")
        
        # 3. Accept/Reject Decision
        if test_loss < best_loss:
            print(f"✅ ACCEPTED! Loss improved from {best_loss:.6f} -> {test_loss:.6f}")
            current_config = proposed_config
            best_loss = test_loss
            baseline_metrics = test_metrics
            
            # Persist the new active config
            with open(CONFIG_FILE, "w") as f:
                json.dump(current_config, f, indent=2)
                
            # PRESERVE THE BRAIN: Copy the weights of the winning configuration!
            if os.path.exists("last_run_weights.pt"):
                shutil.copy("last_run_weights.pt", "best_model_weights.pt")
                
        else:
            print(f"❌ REJECTED! Loss degraded or flatlined ({test_loss:.6f} >= {best_loss:.6f})")
            
        # 4. Log the history context
        cycle_log = {
            "cycle": cycle,
            "rationale": rationale,
            "baseline_loss": best_loss,
            "candidate_loss": test_loss,
            "verdict": "ACCEPTED" if test_loss < best_loss else "REJECTED",
            "proposed_config": proposed_config
        }
            
        history.append(cycle_log)
        
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
        
        # Cleanup temp file
        if os.path.exists("proposed_config.json"):
            os.remove("proposed_config.json")
            
        time.sleep(1) # Small pause before next cycle
        

if __name__ == "__main__":
    run_operation_evolve()
