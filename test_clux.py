
import sys
import clux

print("[✓] Successfully imported 'clux' native module!")
print("[*] Loading best_model.bin into Python...")

try:
    # Rust-ல் உருவாக்கிய மாடலை Python-ல் Object ஆக ஏற்றுகிறோம்!
    model = clux.SovereignModel("/kaggle/working/CLUX/best_model.bin")
    
    prompts = ["அகர", "கற்க", "துப்பார்க்கு"]
    for p in prompts:
        # Prompt, Max_Tokens, Temperature, Top_K
        out = model.generate(p, 40, 0.7, 5)
        print(f"\nPrompt: '{p}'")
        print(f"Output: {out}")
        
except Exception as e:
    print(f"[!] Python Error: {e}")
