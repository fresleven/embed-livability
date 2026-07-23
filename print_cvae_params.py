"""
Script to print out CVAE parameters identified from the model.
This helps verify which parameters will be optimized by the CVAE optimizer.
"""

import torch
import argparse
from MMBT_liva.image_liva import CVAEEncoder, CVAEDecoder

# Mock model structure for demonstration
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Simulate CVAE components
        self.image_encoder = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
        )
        self.cvae_encoder = torch.nn.Linear(50, 32)
        self.cvae_decoder = torch.nn.Linear(32, 50)
        
        # Simulate MMBT components
        self.text_encoder = torch.nn.Linear(100, 50)
        self.fusion_layer = torch.nn.Linear(100, 10)

def print_cvae_params(model):
    """
    Print out all CVAE and MMBT parameters from the model.
    """
    cvae_params = []
    mmbt_params = []
    
    print("=" * 80)
    print("PARAMETER ANALYSIS")
    print("=" * 80)
    
    for n, p in model.named_parameters():
        if "cvae" in n.lower():
            cvae_params.append((n, p))
        else:
            mmbt_params.append((n, p))
    
    print(f"\n{'CVAE PARAMETERS':^80}")
    print("-" * 80)
    print(f"{'Parameter Name':<60} {'Shape':>15}")
    print("-" * 80)
    
    total_cvae_params = 0
    for name, param in cvae_params:
        shape_str = str(list(param.shape))
        num_params = param.numel()
        total_cvae_params += num_params
        print(f"{name:<60} {shape_str:>15}")
    
    print("-" * 80)
    print(f"{'Total CVAE parameters:':<60} {total_cvae_params:>15,}")
    
    print(f"\n{'MMBT PARAMETERS':^80}")
    print("-" * 80)
    print(f"{'Parameter Name':<60} {'Shape':>15}")
    print("-" * 80)
    
    total_mmbt_params = 0
    for name, param in mmbt_params:
        shape_str = str(list(param.shape))
        num_params = param.numel()
        total_mmbt_params += num_params
        print(f"{name:<60} {shape_str:>15}")
    
    print("-" * 80)
    print(f"{'Total MMBT parameters:':<60} {total_mmbt_params:>15,}")
    
    print(f"\n{'SUMMARY':^80}")
    print("-" * 80)
    print(f"CVAE params:  {len(cvae_params):>5} groups | {total_cvae_params:>10,} total parameters")
    print(f"MMBT params:  {len(mmbt_params):>5} groups | {total_mmbt_params:>10,} total parameters")
    print(f"Total params: {len(cvae_params) + len(mmbt_params):>5} groups | {total_cvae_params + total_mmbt_params:>10,} total parameters")
    print("=" * 80)
    
    return cvae_params, mmbt_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print CVAE parameters from model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    
    # Use mock model for demonstration
    print("\nUsing mock model for demonstration.")
    print("To use with an actual model, pass --model_path to the checkpoint directory.\n")
    
    model = MockModel()
    cvae_params, mmbt_params = print_cvae_params(model)
