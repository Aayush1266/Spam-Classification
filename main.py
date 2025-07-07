import torch
import tiktoken
from GPTModel import GPTModel
from setup import BASE_CONFIG
from utility_functions import classify_review

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_and_optimizer = torch.load("classifier-model-and-optimizer.pth")
model = GPTModel(BASE_CONFIG)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
model.load_state_dict(model_and_optimizer["model"])
model.eval()

email = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."

print(classify_review(email, model, tokenizer, device,max_length=len(email)))