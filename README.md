# Ovis: Structural Embedding Alignment for Multimodal Large Language Model

Ovis is a novel Multimodal Large Language Model (MLLM) architecture, designed to structurally align visual and textual embeddings. For a comprehensive introduction, please refer to the [Ovis paper](https://arxiv.org/abs/2405.20797).

<div style="text-align: center;">
  <img style="max-width: 100%;" src="docs/ovis-illustration.png" alt="Ovis Illustration"/>
</div>

## Release
- [06/14] 🔥 We release the [paper](https://arxiv.org/pdf/2405.20797), [code](https://github.com/AIDC-AI/Ovis), [models](https://huggingface.co/AIDC-AI/Ovis-Clip-Llama3-8B), and [datasets](https://huggingface.co/datasets/AIDC-AI/Ovis-dataset).

## Contents
- [Install](#install)
- [Model](#model)
- [Performance](#performance)
- [Dataset](#dataset)
- [Train](#train)
- [Inference](#inference)
- [Citation](#citation)
- [Team](#team)
- [License](#license)

## Install
Ovis has been tested with Python 3.10, Torch 2.1.0, Transformers 4.41.1, and DeepSpeed 0.14.0. For a comprehensive list of package dependencies, please consult the `requirements.txt` file. Before training or inference, please install Ovis as follows.
```bash
git clone git@github.com:AIDC-AI/Ovis.git
conda create -n ovis python=3.10 -y
conda activate ovis
cd Ovis
pip install -r requirements.txt
pip install -e .
```

## Model
Ovis can be instantiated with popular LLMs (e.g., Qwen, Llama3). We provide the following pretrained Ovis MLLMs:

| Ovis MLLMs            | ViT   | LLM                |                              Download                               | MMStar | MMB-EN | MMB-CN | MMMU-Val | MMMU-Test | MathVista-Mini |  MME | HallusionBench | RealWorldQA | 
|:----------------------|:------|:-------------------|:-------------------------------------------------------------------:|-------:|-------:|-------:|---------:|----------:|---------------:|-----:|---------------:|------------:|
| Ovis-Clip-Qwen1.5-7B  | Clip  | Qwen1.5-7B-Chat    | [Huggingface](https://huggingface.co/AIDC-AI/Ovis-Clip-Qwen1_5-7B)  |   44.3 |   75.1 |   70.2 |     39.7 |      37.7 |           41.4 | 1882 |           56.4 |        60.0 |
| Ovis-Clip-Llama3-8B   | Clip  | Llama3-8B-Instruct |  [Huggingface](https://huggingface.co/AIDC-AI/Ovis-Clip-Llama3-8B)  |   49.5 |   77.4 |   72.8 |     44.7 |      39.0 |           40.8 | 2009 |           61.1 |        57.9 |
| Ovis-Clip-Qwen1.5-14B | Clip  | Qwen1.5-14B-Chat   | [Huggingface](https://huggingface.co/AIDC-AI/Ovis-Clip-Qwen1_5-14B) |   48.5 |   78.4 |   76.6 |     46.7 |      40.7 |           43.4 | 1961 |           57.6 |        62.7 |

## Performance
We evaluate Ovis across various multimodal benchmarks using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). The evaluation results show that Ovis outperforms open-source MLLMs within the same parameter tier across various benchmarks, and
Ovis-Clip-Qwen1.5-14B also surpasses the high-resource proprietary model Qwen-VL-Plus overall.

<div style="text-align: center;">
  <img style="max-width: 100%;" src="docs/performance.png" alt="Ovis Performance"/>
</div>

## Dataset
All training datasets are summarized in the JSON file located at `ovis/train/dataset_info.json`. Each dataset entry includes the following attributes:

- **`meta_file`**: This file contains a collection of samples where each sample consists of text and (optionally) image. The text data is embedded directly within the `meta_file`, while the image is represented by its filename. This filename refers to the image file located in the `image_dir`.
- **`image_dir`**: The directory where the images are stored.
- **`data_format`**: Specifies the format of the data, which is used to determine the dataset class for processing the dataset.

We provide the `meta_file` for each training dataset at [Huggingface](https://huggingface.co/datasets/AIDC-AI/Ovis-dataset). The images can be downloaded from their respective sources listed below.

| dataset name                   |      image dir |                                                  image source |
|:-------------------------------|---------------:|--------------------------------------------------------------:|
| coyo-10m                       |       coyo_10m |              `image_url` of each sample in `coyo-10m.parquet` |
| llava-pretrain-558k            | llava_pretrain |     https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain |
| sharegpt4v-pretrain-82k        |     sharegpt4v |           https://huggingface.co/datasets/Lin-Chen/ShareGPT4V |
| allava-caption-laion-4v-485k   |   allava_laion | https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V |
| allava-caption-vflan-4v-203k   |   allava_vflan | https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V |
| laion-description-11k          |     ovis_laion |          https://huggingface.co/datasets/AIDC-AI/Ovis-dataset |
| cc12m-description-1m           |     ovis_cc12m |          https://huggingface.co/datasets/AIDC-AI/Ovis-dataset |
| scienceqa-train-val-17k        |      scienceqa |                                   https://scienceqa.github.io |
| textvqa-train-35k              |        textvqa |                                           https://textvqa.org |
| allava-instruct-laion-4v-485k  |   allava_laion | https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V |
| allava-instruct-vflan-4v-203k  |   allava_vflan | https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V |
| arxivqa-100k                   |        arxivqa |         https://huggingface.co/datasets/MMInstruction/ArxivQA |
| q-instruct-198k                |     q_instruct |        https://huggingface.co/datasets/q-future/Q-Instruct-DB |
| llava-finetune-665k            | llava_finetune |                          https://github.com/haotian-liu/LLaVA |
| geo-177k                       |            geo |              https://huggingface.co/datasets/Luckyjhg/Geo170K |
| lrv-and-chart-instruction-343k |  lrv_and_chart |                  https://github.com/FuxiaoLiu/LRV-Instruction |
| synthdog-en-ocr-200k           |       synthdog |    https://huggingface.co/datasets/naver-clova-ix/synthdog-en |
| allava-evol-instruct-143k      |              - |                                                             - |
| cc12m-qa-387k                  |     ovis_cc12m |          https://huggingface.co/datasets/AIDC-AI/Ovis-dataset |

Below is an example of the folder structure consistent with `ovis/train/dataset_info.json`. You can alter the folder structure as needed and modify `ovis/train/dataset_info.json` accordingly.
```
|-- mllm_datasets
    |-- meta_files
        |-- coyo-10m.parquet
        |-- llava-pretrain-558k.json
        |-- sharegpt4v-pretrain-82k.json
        |-- allava-caption-laion-4v-485k.json
        ...
    |-- images
        |-- coyo_10m
        |-- llava_pretrain
        |-- sharegpt4v
        |-- allava_laion
        ...
```

## Train
Ovis is trained in three stages, with each stage's training scripts located in the `scripts` directory. Before starting the training, ensure you properly set the `ROOT` variable in the scripts. Below are the commands to train Ovis-Clip-Qwen1.5-7B:
```bash
bash scripts/v1/Ovis-Clip-Qwen1.5-7B-S1.sh
bash scripts/v1/Ovis-Clip-Qwen1.5-7B-S2.sh
bash scripts/v1/Ovis-Clip-Qwen1.5-7B-S3.sh
```

## Inference
We provide an inference wrapper in `ovis/serve/runner.py`, which can be used as:
```python
from PIL import Image
from ovis.serve.runner import RunnerArguments, OvisRunner
image = Image.open('IMAGE_PATH')
text = 'PROMPT'
runner_args = RunnerArguments(model_path='MODEL_PATH')
runner = OvisRunner(runner_args)
generation = runner.run(image, text)
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
This work is a collaborative effort by the MarcoVL team. We would also like to provide links to the following MLLM papers from our team:
- [Parrot: Multilingual Visual Instruction Tuning](https://arxiv.org/abs/2406.02539)
- [Wings: Learning Multimodal LLMs without Text-only Forgetting](https://arxiv.org/abs/2406.03496)

## License
The project is licensed under the Apache 2.0 License and is restricted to uses that comply with the license agreements of Qwen, Llama3, and Clip.
