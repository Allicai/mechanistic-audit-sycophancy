import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# hugging face hub IDs
MODEL_MAP = {
    "llama-3": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct", 
    "gemma": "google/gemma-2-9b-it"
}

class AuditModel:
    """
    A wrapper around the HF model to simplify hook registration 
    and activation retrieval for mechanistic audits.
    """
    def __init__(self, model_name, device="cuda"):
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            print("Warning: HF_TOKEN not found. Assuming cached login...")
        self.model_name = model_name
        self.device = device
        self.hf_id = MODEL_MAP.get(model_name, model_name)
        
        print(f"Loading {self.hf_id} on {device}...")
        
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # load model (in bfloat16 for efficiency)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True, # needed for models like Qwen/Phi
            token=self.hf_token
        )
        self.model.eval() # eval mode

    def format_prompt(self, messages):
        """
        Applies the correct chat template for the specific model.
        Crucial for sycophancy audits to work correctly.
        """
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def get_layers(self):
        """
        Returns the list of transformer layers to iterate over.
        Abstracts away the naming differences (e.g. model.layers vs model.model.layers).
        """
        if "llama" in self.model_name.lower():
            return self.model.model.layers
        elif "qwen" in self.model_name.lower():
            return self.model.model.layers
        elif "gemma" in self.model_name.lower():
            return self.model.model.layers
        else:
            # fallback
            return self.model.model.layers

    def generate(self, prompt, max_new_tokens=100):
        """
        Simple generation wrapper for sanity checks.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# grab the instance
def load_audit_model(model_key="llama-3"):
    return AuditModel(model_key)

if __name__ == "__main__":
    print("Testing loader...")
    wrapper = load_audit_model("llama-3")
    
    msgs = [{"role": "user", "content": "Hello, are you conscious?"}]
    prompt = wrapper.format_prompt(msgs)
    print(f"\nFormatted Prompt:\n{prompt}")
    
    response = wrapper.generate(prompt, max_new_tokens=20)
    print(f"\nResponse:\n{response}")