import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

import os
import re
import sys
from datetime import datetime
from training.consts import DEFAULT_INPUT_MODEL, SUGGESTED_INPUT_MODELS
from training.trainer import load_training_dataset, load_tokenizer

from training.generate import generate_response, load_model_tokenizer_for_generate

model_path = sys.argv[1]
print("Loading model from path {}".format(model_path))
model, tokenizer = load_model_tokenizer_for_generate(model_path) # Mention the output path here

# COMMAND ----------

instructions = [
    "Explain Decimal Numbers for me"
]

# Use the model to generate responses for each of the instructions above.
for instruction in instructions:
    response = generate_response(instruction, model=model, tokenizer=tokenizer)
    if response:
        print(f"Instruction: {instruction}\n\n{response}\n\n-----------\n")
