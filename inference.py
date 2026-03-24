import sys
import json
import torch
import torch.nn.functional as F
from train_eval import SimpleTransformerLM
import tiktoken

def load_best_model():
    with open("config.json", "r") as f:
        config = json.load(f)
        
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = config.get("vocab_size", 50257)
    
    model = SimpleTransformerLM(config, vocab_size=vocab_size, max_seq_len=128)
    
    try:
        # Load weights, map to CPU initially to ensure compatibility regardless of local device
        model.load_state_dict(torch.load("best_model_weights.pt", map_location=torch.device('cpu'), weights_only=True))
        print("✅ Loaded best_model_weights.pt successfully!")
    except Exception as e:
        print(f"❌ Failed to load weights. Wait for the Controller to find a better configuration first! Error: {e}")
        sys.exit(1)
        
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Encode prompt
    tokens = tokenizer.encode(prompt, allowed_special="all")
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"\nPrompt: {prompt}")
    print("Generating", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to max sequence length supported
            x_cond = x[:, -128:] 
            
            logits = model(x_cond)
            # Focus only on the last time step
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append token to sequence
            x = torch.cat((x, next_token), dim=1)
            
            print(".", end="", flush=True)
            
    print("\n\n-- Generation Complete --\n")
    output_text = tokenizer.decode(x[0].tolist())
    print(output_text)

if __name__ == "__main__":
    prompt = "The future of Artificial Intelligence is"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        
    model, tokenizer = load_best_model()
    generate_text(model, tokenizer, prompt)
