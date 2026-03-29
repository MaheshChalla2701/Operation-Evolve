import torch
import os
import glob
from data import load_dataset
from config import EvolveConfig

def inspect():
    config = EvolveConfig()
    data_dir = config.data_dir
    files = glob.glob(os.path.join(data_dir, "dataset_*.pt"))
    
    if not files:
        print(f"No datasets found in {data_dir}. Run pipelined_orchestrator.py first!")
        return

    print(f"\n{'='*60}")
    print(f"  DATASET INSPECTOR (Checking physical .pt files)")
    print(f"{'='*60}")
    
    for f in sorted(files):
        name = os.path.basename(f)
        try:
            # We use torch.load manually to show raw info or use our loader
            content = torch.load(f, weights_only=True)
            features = content["features"]
            labels = content["labels"]
            
            print(f"\n📂 File: {name}")
            print(f"   - Samples: {features.shape[0]}")
            print(f"   - Feature Dim: {features.shape[1]}")
            print(f"   - Classes present: {labels.unique().tolist()}")
            
            # Show a mean vector for Class 0 to prove it's different across datasets
            class_0_mask = (labels == 0)
            if class_0_mask.any():
                mean_v = features[class_0_mask].mean(dim=0)
                print(f"   - Class 0 'Center' (First 5 dims): {mean_v[:5].tolist()}")
            else:
                print(f"   - Class 0 not found in this batch.")
                
        except Exception as e:
            print(f"   ! Error reading {name}: {e}")
            
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    inspect()
