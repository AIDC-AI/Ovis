import torch
from PIL import Image
from ovis.model.modeling_ovis import Ovis

MODEL_PATH = "AIDC-AI/Ovis2.5-9B"

# Enable reflective reasoning mode (thinking mode)
enable_thinking = True

# Total tokens = thinking phase + response
max_new_tokens = 3072

# thinking_budget: upper bound of tokens reserved for the "thinking phase"
# - If provided, the model will stop thinking once this budget is reached,
#   then switch to generating the final response.
# - If omitted when calling .chat(), it is equivalent to "not set",
#   and the model may use all max_new_tokens for thinking.
thinking_budget = 2048

# Load model
model = Ovis.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
).eval()

prompt = "<image>\nDescribe this image in detail."
images = [Image.open("/path/to/image1.jpg")]

# Run chat
response, thinking, _ = model.chat(
    prompt=prompt,
    images=images,
    history=None,
    do_sample=True,
    max_new_tokens=max_new_tokens,
    enable_thinking=enable_thinking,
    thinking_budget=thinking_budget,  # omit this arg => unlimited thinking
)

# Print results
if enable_thinking and thinking:
    print("=== Thinking ===")
    print(thinking)
    print("\n=== Response ===")
    print(response)
else:
    print("Response:", response)
