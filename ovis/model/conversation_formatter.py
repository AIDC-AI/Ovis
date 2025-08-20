import copy
from abc import ABC, abstractmethod
from typing import List, Dict

from ovis.util.constants import IMAGE_TOKEN_ID, IGNORE_ID, IMAGE_TOKEN, VIDEO_TOKEN_ID, VIDEO_TOKEN


class ConversationFormatter(ABC):
    support_tokenizer_types = None

    def __init__(self, tokenizer):
        tokenizer_type = type(tokenizer).__name__
        assert tokenizer_type in self.support_tokenizer_types, \
            f'Invalid tokenizer type, expected one from `{self.support_tokenizer_types}`, but got `{tokenizer_type}`'
        self.tokenizer = tokenizer
        self.image_token = IMAGE_TOKEN
        self.image_token_id = IMAGE_TOKEN_ID
        self.ignore_id = IGNORE_ID
        self.im_end = None
        self.video_token = VIDEO_TOKEN
        self.video_token_id = VIDEO_TOKEN_ID

    def _tokenize_with_image_symbol(self, text):
        if text.find(self.video_token) != -1:
            token = self.video_token
            token_id = self.video_token_id
        else:
            token = self.image_token
            token_id = self.image_token_id

        text_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in
                       text.split(token)]
        token_ids = []
        num_chuck = len(text_chunks)
        for i, chunk in enumerate(text_chunks):
            token_ids.extend(chunk)
            if i < num_chuck - 1:
                token_ids.append(token_id)
        return token_ids

    @abstractmethod
    def format(self, conversations: List[Dict], generation_preface=None, enable_thinking=False):
        pass

    @abstractmethod
    def format_query(self, query, generation_preface=""):
        pass

class Qwen3ConversationFormatter(ConversationFormatter):
    support_tokenizer_types = ['QWenTokenizer', 'Qwen2TokenizerFast']

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.from2role = {
            "system": "<|im_start|>system\n",
            "human": "<|im_start|>user\n",
            "gpt": "<|im_start|>assistant\n",
            "ignored_gpt": "<|im_start|>assistant\n",
        }
        
        self.im_end = "<|im_end|>\n"
        self.empty_think = "<think>\n\n</think>\n\n"
        self.gpt_token_nums = None

    def _initialize_gpt_token_nums(self) -> Dict[str, int]:
        think_prefix = self.from2role["gpt"]
        think_num = len(
            self.tokenizer(think_prefix, add_special_tokens=False).input_ids
        )
        no_think_prefix = self.from2role["gpt"] + self.empty_think
        no_think_num = len(
            self.tokenizer(no_think_prefix, add_special_tokens=False).input_ids
        )
        return {'think': think_num, 'no_think': no_think_num}

    # enable_thinking is deprecated
    def format(self, conversations: List[Dict], generation_preface=None, enable_thinking=False):
        conversations = copy.deepcopy(conversations)

        if generation_preface is not None:
            conversations.append({
                "from": "gpt",
                "value": generation_preface
            })

        prompt = ""
        input_ids = []
        labels = []
        num_conversation = len(conversations)
        for i, conversation in enumerate(conversations):
            frm = conversation["from"]
            role = self.from2role[frm]
            message = conversation["value"]
            has_thinking = '<think>' in message and '</think>' in message
            if frm == 'gpt' and not has_thinking and generation_preface is None:
                text = role + self.empty_think + message
            else:
                text = role + message
            
            if self.gpt_token_nums is None:
                self.gpt_token_nums = self._initialize_gpt_token_nums()
            gpt_token_num = self.gpt_token_nums['think'] if has_thinking else self.gpt_token_nums['no_think']
            
            if i < num_conversation - 1 or generation_preface is None:
                text += self.im_end
            prompt += text
            token_ids = self._tokenize_with_image_symbol(text)
            input_ids.extend(token_ids)
            label_ids = [self.ignore_id] * len(token_ids)
            if frm == "gpt" and generation_preface is None:
                # learning `\n` following `im_end` is meaningless, so the last `\n` token is ignored in label
                label_ids[gpt_token_num:-1] = token_ids[gpt_token_num:-1]
            labels.extend(label_ids)

        assert self._tokenize_with_image_symbol(prompt) == input_ids
        assert len(input_ids) == len(labels)

        if conversations[-1]['from'] == "gpt" and generation_preface is None:
            # remove the last `\n` following `im_end` in input_ids
            input_ids.pop()
            labels.pop()

        return prompt, input_ids, labels

    def format_query(self, query, generation_preface="", enable_thinking=False):
        prompt, input_ids, _ = self.format([{
            "from": "human",
            "value": query
        }], generation_preface=generation_preface, enable_thinking=enable_thinking)

        return prompt, input_ids
