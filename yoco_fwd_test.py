from fms_fsdp.utils.config_utils import get_model_config
from fms.models.llama_yoco import LLaMA
from transformers import AutoTokenizer
import torch

llama_config = get_model_config("llama_1b")
model = LLaMA(llama_config)
model.reset_parameters()
model.half().cuda()

print(model)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
input_ids = tokenizer("What is your favorite TV show?", return_tensors="pt").input_ids.to(next(model.parameters()).device)

with torch.no_grad():
    output = model(input_ids)

print(output)

# test 2, use previously saved data and labels to calc loss
# zl_coeff: float = 1e-4
# exam_inp = torch.load("exam_inp.pt")
# for batch in exam_inp:
#     inputs = batch["inputs"].unsqueeze(0).to("cuda")  # use one batch of data, keep the batch dim
#     label = batch["labels"].unsqueeze(0).to("cuda")

#     output = model(inputs)
#     ce_loss = torch.nn.CrossEntropyLoss()
#     loss = ce_loss(output.view(-1, output.size(-1)), label.view(-1).long())
#     loss = loss + zl_coeff * torch.logsumexp(output, dim=-1).pow(2).mean()
#     print(loss)
# print(model.clip_grad_norm_(1.0).item())
