import torch
from PIL import Image
from ovis.model.modeling_ovis import Ovis

# If you need video support, make sure moviepy is installed first:
#   pip install moviepy==1.0.3
try:
    from moviepy.editor import VideoFileClip  # type: ignore
    _HAS_MOVIEPY = True
except Exception:
    _HAS_MOVIEPY = False


def run_single_image_example(model: Ovis, image_path: str) -> None:
    """
    Run an inference example with a single image input.
    """
    print("--- 1) Single-image example ---")
    images = [Image.open(image_path).convert("RGB")]
    prompt = "<image>\nDescribe this image in detail."

    print(f"Prompt:\n{prompt}")

    response, _, _ = model.chat(
        prompt=prompt,
        images=images,
        min_pixels=448 * 448,
        max_pixels=1792 * 1792,
        videos=None,
        do_sample=True,
        max_new_tokens=1024,
    )
    print(f"\nResponse:\n{response}")


def run_multi_image_example(model: Ovis, image_paths: list) -> None:
    """
    Run an inference example with multiple image inputs.
    """
    print("--- 2) Multi-image example ---")
    images = [Image.open(p).convert("RGB") for p in image_paths]
    prompt = "<image>\n<image>\n<image>\nWhat is the relationship between the third image and the first two?"

    print(f"Prompt:\n{prompt}")

    response, _, _ = model.chat(
        prompt=prompt,
        images=images,
        min_pixels=448 * 448,
        max_pixels=896 * 896,
        videos=None,
        do_sample=True,
        max_new_tokens=1024,
    )
    print(f"\nResponse:\n{response}")


def run_video_example(model: Ovis, video_path: str, num_frames: int = 8) -> None:
    """
    Run an inference example with a video input.
    """
    if not _HAS_MOVIEPY:
        raise ImportError(
            "moviepy is not installed. Install it with `pip install moviepy==1.0.3` to use video examples."
        )

    print("--- 3) Video example ---")
    with VideoFileClip(video_path) as clip:
        total_frames = int(clip.fps * clip.duration)
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frames = [
            Image.fromarray(clip.get_frame(t)) for t in (index / clip.fps for index in indices)
        ]

    videos = [frames]
    prompt = "<video>\nDescribe this video in detail."

    print(f"Prompt:\n{prompt}")

    response, _, _ = model.chat(
        prompt=prompt,
        images=None,
        videos=videos,
        min_pixels=448 * 448,
        max_pixels=896 * 896,
        do_sample=True,
        max_new_tokens=1024,
    )
    print(f"\nResponse:\n{response}")


def run_text_only_example(model: Ovis) -> None:
    """
    Run an inference example with text-only input.
    """
    print("--- 4) Text-only example ---")
    prompt = "Hi, please introduce Huangshan (Yellow Mountain) in Chinese."

    print(f"Prompt:\n{prompt}")

    response, _, _ = model.chat(
        prompt=prompt,
        images=None,
        videos=None,
        do_sample=True,
        max_new_tokens=1024,
    )
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    # --- 1) Load model ---
    model_path = "AIDC-AI/Ovis2.5-9B"

    print("Loading model, please wait...")
    model = (
        Ovis.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda:0",
        ).eval()
    )
    print("Model loaded.")
    print("\n========================================\n")

    # --- 2) Define file paths (anonymized placeholders) ---
    # Replace the following with your own paths
    single_image_file = "/path/to/image1.jpg"
    multi_image_files = [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg",
        "/path/to/image3.png",
    ]
    video_file = "/path/to/video1.mp4"

    # --- 3) Run examples ---
    run_single_image_example(model, single_image_file)
    print("\n========================================\n")

    run_multi_image_example(model, multi_image_files)
    print("\n========================================\n")

    run_video_example(model, video_file)
    print("\n========================================\n")

    run_text_only_example(model)
    print("\n========================================\n")
