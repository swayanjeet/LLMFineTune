
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

import os
import re
from datetime import datetime
from training.consts import DEFAULT_INPUT_MODEL, SUGGESTED_INPUT_MODELS
from training.trainer import load_training_dataset, load_tokenizer

# Cache data and tokenizer locally before creating a bunch of deepspeed processes and make sure they succeeds.
load_training_dataset()
load_tokenizer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
