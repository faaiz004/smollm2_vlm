import numpy as np
from tqdm import tqdm
import torch


# This will be used to calculate the perplexity

def compute_perplexity(smol_lm, smol_tokenizer, alpaca_dataset):

  smol_lm.eval()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  smol_lm.to(device)


  losses = []
  for example in tqdm(alpaca_dataset, desc = "Computing PPL"):
    # this is like the standard structure that apply message template needs and then it maps it to specific structure of different models.
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = smol_tokenizer.apply_chat_template(messages, tokenize = False)
    inputs = smol_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    # label = inputs["input_ids"] this specifies where to compute the loss, right now loss is being computed on the input
    with torch.no_grad():
      outputs = smol_lm(**inputs, labels = inputs["input_ids"])
    loss = outputs.loss
    losses.append(loss.item())

  ppl_0 = np.exp(np.mean(losses))
  print("\n")
  print(f"PPL_0 (baseline perplexity): {ppl_0:.2f}")
  return ppl_0



