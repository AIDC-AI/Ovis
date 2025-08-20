import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# for torch==2.4.0
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint", lineno=1399)
