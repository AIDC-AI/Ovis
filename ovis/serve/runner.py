from dataclasses import field, dataclass

import torch
from PIL import Image

from ovis.model.modeling_ovis import Ovis
from ovis.util.constants import IMAGE_TOKEN


@dataclass
class RunnerArguments:
    model_path: str
    do_sample: bool = field(default=False)
    temperature: float = field(default=1.0)
    top_p: float = field(default=None)
    num_beams: int = field(default=1)
    max_new_tokens: int = field(default=512)


class OvisRunner:
    def __init__(self, args: RunnerArguments):
        self.args = args
        self.model = Ovis.from_pretrained(self.args.model_path,
                                          torch_dtype=torch.bfloat16,
                                          multimodal_max_length=8192).cuda()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.conversation_formatter = self.model.get_conversation_formatter()

    def process_input(self, image, text):
        query = f'{IMAGE_TOKEN} {text}'
        prompt, input_ids = self.conversation_formatter.format_query(query)
        input_ids = torch.unsqueeze(input_ids, dim=0)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        pixel_values = self.visual_tokenizer.preprocess_image(image)

        return prompt, pixel_values, input_ids, attention_mask

    def run(self, image: Image.Image, text: str, **gen_args):
        prompt, pixel_values, input_ids, attention_mask = self.process_input(image, text)
        input_ids = input_ids.to(device=self.model.device)
        attention_mask = attention_mask.to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype,
                                        device=self.visual_tokenizer.device)]
        with torch.inference_mode():
            kwargs = dict(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                do_sample=self.args.do_sample,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                repetition_penalty=None,
                use_cache=True,
                eos_token_id=self.text_tokenizer.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id
            )
            if self.args.do_sample:
                kwargs["temperature"] = self.args.temperature,
                kwargs["top_p"] = self.args.top_p
            output_ids = self.model.generate(input_ids, **kwargs)[0]
        input_token_len = input_ids.shape[1]
        output_token_len = output_ids.shape[0]
        n_diff_input_output = (input_ids[0] != output_ids[:input_token_len]).sum().item()
        assert n_diff_input_output == 0, f"{n_diff_input_output} output_ids is not the same as the input_ids"
        output = self.text_tokenizer.decode(output_ids[input_token_len:], skip_special_tokens=True)
        response = dict(
            prompt=prompt,
            output=output,
            prompt_tokens=input_token_len,
            total_tokens=output_token_len
        )
        return response
