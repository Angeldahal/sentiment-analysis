import transformers

from models.engine import Transformer

model_ckpt = "bert-base-uncased"

transformer = transformers.AutoModel.from_pretrained(model_ckpt)

output_dim = 2
freeze = False

model = Transformer(transformer, output_dim, freeze)