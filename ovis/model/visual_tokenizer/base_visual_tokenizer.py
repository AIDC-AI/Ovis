from typing import Union, Optional

import PIL.Image
import torch
from torch.nn.functional import softmax, gumbel_softmax
from transformers import PretrainedConfig, PreTrainedModel, AutoImageProcessor, AutoModel, AutoConfig


class BaseVisualTokenizerConfig(PretrainedConfig):
    def __init__(self,
                 vocab_size=16384,
                 tokenize_function="softmax",
                 tau=1.0,
                 depths=None,
                 use_indicators=False,
                 drop_cls_token=False,
                 backbone_config: Optional[Union[PretrainedConfig, dict]] = None,
                 hidden_stride: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.tokenize_function = tokenize_function
        self.tau = tau
        if isinstance(depths, str):
            depths = [int(x) for x in depths.split('|')]
        self.depths = depths
        self.backbone_kwargs = {}
        self.use_indicators = use_indicators
        self.drop_cls_token = drop_cls_token
        if backbone_config is not None:
            assert isinstance(backbone_config, (PretrainedConfig, dict)), \
                f"expect `backbone_config` to be instance of PretrainedConfig or dict, but got {type(backbone_config)} type"
            if not isinstance(backbone_config, PretrainedConfig):
                model_type = backbone_config['model_type']
                backbone_config.pop('model_type')
                backbone_config = AutoConfig.for_model(model_type, **backbone_config)
        self.backbone_config = backbone_config
        self.hidden_stride = hidden_stride


class BaseVisualTokenizer(PreTrainedModel):
    base_model_prefix = "backbone"
    main_input_name = None
    _image_processor_class = None
    _image_processor_kwargs = {}
    _backbone_class = None
    _backbone_name_or_path = None

    def __init__(self, config: BaseVisualTokenizerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if kwargs.get('train_from_scratch'):
            self.image_processor = self._image_processor_class.from_pretrained(self._backbone_name_or_path,
                                                                               **self._image_processor_kwargs)
            self.backbone = self._backbone_class.from_pretrained(self._backbone_name_or_path,
                                                                 **self.config.backbone_kwargs)
            self.config.backbone_config = self.backbone.config
        else:
            self.image_processor = AutoImageProcessor.from_pretrained(kwargs['image_processor_name_or_path'])
            self.backbone = AutoModel.from_config(self.config.backbone_config)
        self.head = None

        assert all((self.image_processor.do_resize,
                    not getattr(self.image_processor, 'do_center_crop', False),
                    self.image_processor.do_rescale,
                    self.image_processor.do_normalize
                    )), f"image_processor `{self.image_processor}` is not supported currently"

    def get_backbone(self):
        return self.backbone

    def get_monitor_tensors(self):
        raise NotImplementedError

    def get_image_processor(self):
        return self.image_processor

    def get_head(self):
        return self.head

    def get_image_size(self):
        raise NotImplementedError

    def preprocess_image(self, image: PIL.Image.Image, convert_to_rgb=True):
        if convert_to_rgb and image.mode != 'RGB':
            image = image.convert('RGB')

        # first resize and preprocess
        sides = self.get_image_size()
        if sides[0] != sides[1]:
            raise ValueError('get_image_size() returns non-square size')
        side = sides[0]

        width, height = image.size
        if width == height:
            new_width = new_height = side
        elif width > height:
            new_width = side
            new_height = int(height / width * new_width)
        else:
            new_height = side
            new_width = int(width / height * new_height)
        new_size = dict(height=new_height, width=new_width)
        pixel_values = self.image_processor.preprocess(image, size=new_size, return_tensors='pt')['pixel_values']

        # then pad to square
        square_values = torch.zeros([1, 3, side, side], dtype=pixel_values.dtype, device=pixel_values.device)
        new_height, new_width = pixel_values.shape[2:]
        if new_height == new_width:
            square_values[:, :, :, :] = pixel_values
        elif new_height > new_width:
            from_index = (side - new_width) // 2
            square_values[:, :, :, from_index:from_index + new_width] = pixel_values
        else:
            from_index = (side - new_height) // 2
            square_values[:, :, from_index:from_index + new_height, :] = pixel_values

        return square_values

    def get_layer_norm(self):
        return self.layer_norm

    def tokenize(self, logits):
        def st_argmax(y_soft, dim):  # straight-through softmax
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            return ret

        if self.config.tokenize_function == 'softmax':
            tokens = softmax(logits, dim=-1)
        elif self.config.tokenize_function == 'gumbel_argmax':
            tokens = gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.config.tokenize_function == 'st_argmax':
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                f'Invalid `max_type`, expected softmax or gumbel_argmax or st_argmax, but got {self.config.tokenize_function}')
        return tokens
