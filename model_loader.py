from transformers import CLIPModel
import torch
# loading the smolLM model
from transformers import AutoModelForCausalLM, AutoTokenizer



def load_model(full_train, train_indices, clip_processor, clip_eval = True):

  clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  # setting all the gradients to false
  for param in clip_model.parameters():
      param.requires_grad_(False)


  # I have already loaded the processor in cell 1, there is no need to load it again
  if clip_eval:
    clip_model.eval()

  # testing the model for one input
  test_input = full_train[train_indices[0]]
  print(test_input)
  # This has a PIL image at 0,
  image = test_input[0]  # Already a PIL Image
  label = test_input[1]  # The class label (0-9)
  print(image.size)       # (32, 32)


  # now i need to process input using clip process to normalize and resize it
  inputs = clip_processor(images=image, return_tensors="pt")
  # return pt means we are returning pytorch tensors

  with torch.no_grad():
    output = clip_model.vision_model(**inputs)
    print(output.last_hidden_state.shape)





  #use automodelforcasuallm to load the smollm model
  # need to specify modelname and the dtype
  smol_lm = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct", torch_dtype=torch.float32)
  smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

  # Now I need to check the dimensionality
  # Confirm dLM = 960
  print(f"Hidden size (dLM): {smol_lm.config.hidden_size}")  # this should be 960

  # Confirm vocab size = 49152
  print(f"Vocab size: {smol_lm.config.vocab_size}")  # this should be 49152
  return (smol_lm, smol_tokenizer, clip_model)


