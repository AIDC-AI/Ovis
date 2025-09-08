This directory contains the file for the usage of Ovis2 in vllm
To run the model in vllm, until the PR will be accepted, one should do 
```python
from ovis_modeling_directory.ovis_modeling import OvisForConditionalGeneration
ModelRegistry.register_model("Ovis", OvisForConditionalGeneration)
llm = LLM(model="AIDC-AI/Ovis2-2B") # or some other ovis model
```