import torch
import json
import os
from model import SparseMoETransformer
from trainer import load_data

def generate_text_from_best_model(prompt_text, max_new_tokens=200, temperature=0.8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading on {device}...")

    # 1. Load the most evolved configuration
    if not os.path.exists("config.json"):
        print("No config.json found! Has controller.py evolved a model yet?")
        return
        
    with open("config.json", "r") as f:
        config = json.load(f)

    # 2. Get Vocabulary and Encode functions
    _, _, vocab_size, stoi, itos = load_data()
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Ensure strict integers for PyTorch sizing to prevent 'empty()' kwargs mismatch
    config["vocab_size"] = int(vocab_size)
    config["n_embd"] = int(config.get("n_embd", 128))
    config["block_size"] = int(config.get("block_size", 128))
    config["n_layer"] = int(config.get("n_layer", 2))
    config["num_experts"] = int(config.get("num_experts", 4))
    config["expert_hidden_dim"] = int(config.get("expert_hidden_dim", 256))

    # 3. Rebuild the Model Architecture based on the JSON
    model = SparseMoETransformer(config).to(device)

    # 4. Load the Evolved State Dict (Weights)
    if not os.path.exists("best_model.pt"):
        print("No best_model.pt found!")
        return
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    # 5. Encode the user prompt into tensor indices
    if prompt_text:
        context = torch.tensor([encode(prompt_text)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with a blank newline

    print(f"\n[Prompt]: {prompt_text}")
    print("--------------------------------------------------")
    
    # 6. Generate continuation!
    with torch.no_grad():
        generated_indices = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature)
        
    print(decode(generated_indices[0].tolist()))
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    # Feel free to change this prompt to test the model's completion!
    user_prompt = "who r u"
    generate_text_from_best_model(user_prompt, max_new_tokens=250)
