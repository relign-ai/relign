from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)

# Load the model (convert from PyTorch to Flax)
model = FlaxAutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    from_pt=True  # Converts PyTorch weights to Flax
)

prompt = "once upon a time in a land far away"

inputs = tokenizer(prompt, return_tensors="np")

gen_outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    temperature=0.7
)

print("gen outputs", gen_outputs)
print("gen outpu type", type(gen_outputs))

