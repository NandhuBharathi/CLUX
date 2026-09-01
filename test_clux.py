
import sys
import clux

print("[*] Successfully imported 'clux' native module!")
print("[*] Loading best_model.bin into Python...")

try:
    model = clux.SovereignModel("/kaggle/working/CLUX/best_model.bin")
    
    prompts = ["அகர", "கற்க"]
    for p in prompts:
        out = model.generate(p, 40, 0.7, 5)
        print(f"\nPrompt: '{p}'")
        print(f"Output: {out}")
        
except Exception as e:
    print(f"Error: {e}")
