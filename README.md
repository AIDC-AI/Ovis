# Ovis: Structural Embedding Alignment for Multimodal Large Language Model

Ovis (Open VISion) is a novel Multimodal Large Language Model (MLLM) architecture, designed to structurally align visual and textual embeddings. For a comprehensive introduction, please refer to the [Ovis paper](https://arxiv.org/abs/2405.20797).

<div style="text-align: center;">
  <img style="max-width: 100%;" src="docs/ovis-illustration.png" alt="Ovis Illustration"/>
</div>

## Release
- [25/03/25] 🔥 Announcing quantized versions of Ovis2 series, covering [Ovis2-2/4/8/16/34B](https://huggingface.co/AIDC-AI/Ovis2-34B-GPTQ-Int4)!
- [25/01/26] 🔥 Launch of [Ovis2-1/2/4/8/16/34B](https://huggingface.co/AIDC-AI/Ovis2-34B), the latest version of Ovis models, featuring breakthrough small-model performance, enhanced reasoning capabilities, advanced video and multi-image processing, expanded multilingual OCR support, and improved high-resolution image handling.
- [24/11/26] 🔥 Announcing [Ovis1.6-Gemma2-27B](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-27B)!
- [24/11/04] 🔥 Announcing quantized versions of Ovis1.6: [Ovis1.6-Gemma2-9B-GPTQ-Int4](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-9B-GPTQ-Int4) and [Ovis1.6-Llama3.2-3B-GPTQ-Int4](https://huggingface.co/AIDC-AI/Ovis1.6-Llama3.2-3B-GPTQ-Int4)!
- [24/10/22] 🔥 Announcing Ovis1.6-Llama3.2-3B ([Model](https://huggingface.co/AIDC-AI/Ovis1.6-Llama3.2-3B), [Demo](https://huggingface.co/spaces/AIDC-AI/Ovis1.6-Llama3.2-3B))!
- [24/09/19] 🔥 Announcing Ovis1.6-Gemma2-9B ([Model](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-9B), [Demo](https://huggingface.co/spaces/AIDC-AI/Ovis1.6-Gemma2-9B))! This release further enhances high-resolution image processing, is trained on a larger, more diverse, and higher-quality dataset, and refines the training process with DPO training following instruction-tuning.
- [24/07/24] 🔥 Introducing Ovis1.5, featuring improved high-resolution image processing and optimized training data for enhanced performance.
- [24/06/14] 🔥 Launch of Ovis1.0, the inaugural version of the Ovis model.

## Contents
- [Install](#install)
- [Model](#model)
- [Performance](#performance)
- [Inference](#inference)
- [Citation](#citation)
- [Team](#team)
- [License](#license)

## Install
Ovis has been tested with Python 3.10, Torch 2.4.0, Transformers 4.46.2, and DeepSpeed 0.15.4. For a comprehensive list of package dependencies, please consult the `requirements.txt` file.
```bash
git clone git@github.com:AIDC-AI/Ovis.git
conda create -n ovis python=3.10 -y
conda activate ovis
cd Ovis
pip install -r requirements.txt
pip install -e .
```

## Model
Ovis can be instantiated with popular LLMs. We provide the following Ovis MLLMs:

| Ovis MLLMs |           ViT           |          LLM          |                      Model Weights                      |                           Demo                           |
|:-----------|:-----------------------:|:---------------------:|:-------------------------------------------------------:|:--------------------------------------------------------:|
| Ovis2-1B   | aimv2-large-patch14-448 | Qwen2.5-0.5B-Instruct | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-1B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-1B)  |
| Ovis2-2B   | aimv2-large-patch14-448 | Qwen2.5-1.5B-Instruct | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-2B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-2B)  |
| Ovis2-4B   | aimv2-huge-patch14-448  |  Qwen2.5-3B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-4B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-4B)  |
| Ovis2-8B   | aimv2-huge-patch14-448  |  Qwen2.5-7B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-8B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-8B)  |
| Ovis2-16B  | aimv2-huge-patch14-448  | Qwen2.5-14B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-16B) | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-16B) |
| Ovis2-34B  |  aimv2-1B-patch14-448   | Qwen2.5-32B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-34B) |                            -                             |
| Ovis2-2B-GPTQ-Int4   | aimv2-large-patch14-448 | Qwen2.5-1.5B-Instruct | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-2B-GPTQ-Int4)  | -  |
| Ovis2-4B-GPTQ-Int4   | aimv2-huge-patch14-448  |  Qwen2.5-3B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-4B-GPTQ-Int4)  | -  |
| Ovis2-8B-GPTQ-Int4   | aimv2-huge-patch14-448  |  Qwen2.5-7B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-8B-GPTQ-Int4)  | -  |
| Ovis2-16B-GPTQ-Int4  | aimv2-huge-patch14-448  | Qwen2.5-14B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-16B-GPTQ-Int4) | - |
| Ovis2-34B-GPTQ-Int4  |  aimv2-1B-patch14-448   | Qwen2.5-32B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-34B-GPTQ-Int4) | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-34B-GPTQ-Int4) |
| Ovis2-34B-GPTQ-Int8  |  aimv2-1B-patch14-448   | Qwen2.5-32B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-34B-GPTQ-Int8) |                            -                             |

## Performance

![performance-Ovis2](docs/performance/Ovis2.png)

|Benchmark|Ovis2-1B|Ovis2-2B|Ovis2-4B|Ovis2-8B|Ovis2-16B|Ovis2-34B|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|MMBench-V1.1<sub>test</sub>|68.5|77.2|81.4|83.3|85.2|86.2|
|MMStar|52.0|59.0|61.7|64.4|66.9|69.4|
|MMMU<sub>val</sub>|36.0|45.3|48.0|59.0|59.6|65.6|
|MathVista<sub>testmini</sub>|59.5|64.4|69.1|71.4|74.9|77.0|
|HallBench<sub>avg</sub>|44.5|50.2|54.0|56.0|55.9|58.8|
|AI2D<sub>test</sub>|76.8|82.6|85.5|86.8|86.1|88.4|
|OCRBench|88.7|87.5|91.0|89.3|88.2|89.8|
|MMVet|50.3|58.6|65.5|68.5|68.4|75.5|
|Average|59.5|65.6|69.5|72.3|73.1|76.3|

## Inference
We provide an inference wrapper in `ovis/serve/runner.py`, which can be used as:
```python
from PIL import Image
from ovis.serve.runner import RunnerArguments, OvisRunner
image = Image.open('IMAGE_PATH')
text = 'PROMPT'
runner_args = RunnerArguments(model_path='MODEL_PATH')
runner = OvisRunner(runner_args)
generation = runner.run([image, text])
```
Based on [Gradio](https://github.com/gradio-app/gradio), Ovis can also be accessed via a web user interface:
```bash
python ovis/serve/server.py --model_path MODEL_PATH --port PORT
```

## Citation
If you find Ovis useful, please cite the paper
```
@article{lu2024ovis,
  title={Ovis: Structural Embedding Alignment for Multimodal Large Language Model}, 
  author={Shiyin Lu and Yang Li and Qing-Guo Chen and Zhao Xu and Weihua Luo and Kaifu Zhang and Han-Jia Ye},
  year={2024},
  journal={arXiv:2405.20797}
}
```

## Team
This work is a collaborative effort by the Alibaba Ovis team. We would also like to provide links to the following MLLM papers from our team:
- [Parrot: Multilingual Visual Instruction Tuning](https://arxiv.org/abs/2406.02539)
- [Wings: Learning Multimodal LLMs without Text-only Forgetting](https://arxiv.org/abs/2406.03496)

## License
This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) (SPDX-License-Identifier: Apache-2.0).

## Disclaimer
We used compliance-checking algorithms during the training process, to ensure the compliance of the trained model to the best of our ability. Due to the complexity of the data and the diversity of language model usage scenarios, we cannot guarantee that the model is completely free of copyright issues or improper content. If you believe anything infringes on your rights or generates improper content, please contact us, and we will promptly address the matter.
