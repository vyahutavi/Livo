import torch
from data.tokenizer import livorator

def main():
    print("Initializing tokenizer...")
    tokenizer = livorator(verbose=True)
    
    text = "Hello LIVO!"
    print(f"\nEncoding text: '{text}'")
    
    # Generate binary hash features (runs on standard CPU/CUDA)
    spikes = tokenizer.encode_to_spikes(text, spike_dim=10, threshold=0.5)
    
    print("\nBinary Hash Tensor Shape:", spikes.shape)
    print("Format: (batch_size, sequence_length, feature_dim)")
    
    print("\nFirst 5 tokens' binary feature vectors (dim=10):")
    # Show just the first 5 tokens to keep output readable
    for i in range(5):
        print(f"Token {i}: {spikes[0, i, :].tolist()}")
        
if __name__ == "__main__":
    main()
