# adding lora params to smol lm
from peft import get_peft_model, LoraConfig
from connector import Connector
# r = rank of the matrices
# alpha/r determines the weight given to the lora pathway

#Reload fresh SmolLM2 + apply LoRA
#Reload Phase 1 connector weights
#Train with that λ
#Compute PPL and VQA accuracy
#Save weights

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import copy
import gc
import numpy as np # Adding np since it's used in perplexity
import torch
import os


def phase_2_training_config(smol_lm):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    smol_lm = get_peft_model(smol_lm, lora_config) #automatically sets lora param grad = True and other param = False
    smol_lm.print_trainable_parameters()  # Should be < 1%
    return smol_lm



# VQA Dataset class
class VQADataset(Dataset):
    def __init__(self, vqa_data, full_dataset, processor, tokenizer):
        self.vqa_data = vqa_data
        self.full_dataset = full_dataset
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.vqa_data)

    def __getitem__(self, idx):
        item = self.vqa_data[idx]
        pil_image = self.full_dataset[item["image_idx"]][0]
        pixel_values = self.processor(images=pil_image, return_tensors="pt")["pixel_values"].squeeze(0)
        question_ids = self.tokenizer(item["question"], return_tensors="pt")["input_ids"].squeeze(0)
        answer_ids = self.tokenizer(item["answer"], return_tensors="pt")["input_ids"].squeeze(0)
        return pixel_values, question_ids, answer_ids

    def vqa_collate_fn(self, batch):
        pixel_values = torch.stack([item[0] for item in batch])
        question_ids = torch.nn.utils.rnn.pad_sequence(
            [item[1] for item in batch], batch_first=True, padding_value=self.tokenizer.eos_token_id
        )
        answer_ids = torch.nn.utils.rnn.pad_sequence(
            [item[2] for item in batch], batch_first=True, padding_value=self.tokenizer.eos_token_id
        )
        return pixel_values, question_ids, answer_ids

# Alpaca dataset class
class AlpacaDataset(Dataset):
    def __init__(self, alpaca_subset, tokenizer):
        self.data = alpaca_subset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        messages = [
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["output"]},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)["input_ids"].squeeze(0)
        return ids

    def alpaca_collate_fn(self, batch):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=self.tokenizer.eos_token_id)


# Function to compute perplexity
def compute_perplexity(model, alpaca_data, tokenizer, device, n_samples=None):
    model.eval()
    losses = []
    data = alpaca_data.select(range(n_samples)) if n_samples else alpaca_data
    with torch.no_grad():
        for example in data:
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())
    return np.exp(np.mean(losses))


# Function to compute VQA accuracy using Logit-Based Evaluation
def compute_vqa_accuracy(model, connector, clip_model, vqa_val_data, full_test, processor, tokenizer, device, show_progress=False):
    model.eval()
    connector.eval()
    correct = 0
    total = 0
    bos_id = torch.tensor([[tokenizer.bos_token_id]]).to(device)

    # Apply tqdm if show_progress is True
    iterable = tqdm(vqa_val_data, desc="Full VQA Eval") if show_progress else vqa_val_data

    with torch.no_grad():
        for item in iterable:
            pil_image = full_test[item["image_idx"]][0]
            pixel_values = processor(images=pil_image, return_tensors="pt")["pixel_values"].to(device)

            clip_out = clip_model.vision_model(pixel_values=pixel_values)
            patch_features = clip_out.last_hidden_state[:, 1:, :]
            projected = connector(patch_features)
            bos_emb = model.get_input_embeddings()(bos_id)

            question_ids = tokenizer(item["question"], return_tensors="pt")["input_ids"].to(device)
            question_embeds = model.get_input_embeddings()(question_ids)

            inputs_embeds = torch.cat([bos_emb, projected, question_embeds], dim=1)
            expected_ids = tokenizer(
                item["answer"], add_special_tokens=False
            )["input_ids"]
            generated_ids = []
            current_embeds = inputs_embeds
            for _ in range(len(expected_ids) + 1):
                output = model(inputs_embeds=current_embeds)
                predicted_token_id = output.logits[:, -1, :].argmax(dim=-1)
                token_id = predicted_token_id.item()
                if token_id == tokenizer.eos_token_id:
                    break
                generated_ids.append(token_id)
                next_embed = model.get_input_embeddings()(predicted_token_id.unsqueeze(0))
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)
                if len(generated_ids) > len(expected_ids):
                    break

            if expected_ids and generated_ids == expected_ids:
                correct += 1
            total += 1

    return correct / total


# We do quick eval for intermediate checks
def quick_eval(model, connector, clip_model, vqa_val, full_test, alpaca_subset,
               processor, tokenizer, device):
    # Quick VQA on 200 samples (Silent, show_progress=False by default)
    vqa_acc = compute_vqa_accuracy(model, connector, clip_model, vqa_val[:200],
                                    full_test, processor, tokenizer, device)
    # Quick PPL on 100 Alpaca samples
    ppl = compute_perplexity(model, alpaca_subset, tokenizer, device, n_samples=100)

    model.train()
    connector.train()
    return vqa_acc, ppl





# This is the main training function for one λ, we call it again and again
def train_phase2(lambda_val, vqa_train, alpaca_subset, vqa_val, full_train, full_test,
                 processor, tokenizer, clip_model, ppl_0, device, r_norm, epochs=1, batch_size=8, accum_steps=16):

    gc.collect()
    torch.cuda.empty_cache()

    # Reload fresh SmolLM2 + apply LoRA
    print(f"\n{'='*60}\nTraining with λ = {lambda_val}\n{'='*60}")
    # Using float32
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct", torch_dtype=torch.float32).to(device)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    # Reload connector
    fresh_connector = Connector().to(device)
    #This means we will have to first save the connector at this path. 
    fresh_connector.load_state_dict(torch.load("weights/connector_phaseA1.pt", map_location=device))
    fresh_connector.scale_factor = 1.0 / r_norm

    # Datasets and loaders
    vqa_dataset = VQADataset(vqa_train, full_train, processor, tokenizer)
    vqa_loader = DataLoader(vqa_dataset, batch_size=batch_size, shuffle=True, collate_fn=vqa_dataset.vqa_collate_fn)

    alpaca_ds = AlpacaDataset(alpaca_subset, tokenizer)
    alpaca_loader = DataLoader(alpaca_ds, batch_size=batch_size, shuffle=True, collate_fn=alpaca_ds.alpaca_collate_fn)
    alpaca_iter = iter(alpaca_loader)

    # Optimizer over LoRA + connector params
    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(fresh_connector.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    total_steps = len(vqa_loader) * epochs // accum_steps
    scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps, pct_start=0.1)

    clip_model.eval()
    bos_id = torch.tensor([[tokenizer.bos_token_id]]).to(device)

    # Arrays to store step-by-step losses for .npz saving
    vqa_loss_history = []
    lm_loss_history = []
    combined_loss_history = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        fresh_connector.train()
        epoch_losses = []

        optimizer.zero_grad()

        # The main training progress bar
        pbar = tqdm(vqa_loader, desc=f"λ={lambda_val} Ep{epoch+1}")

        for step, (pixel_values, question_ids, answer_ids) in enumerate(pbar):
            pixel_values = pixel_values.to(device)
            question_ids = question_ids.to(device)
            answer_ids = answer_ids.to(device)
            bs = pixel_values.shape[0]
            with torch.no_grad():
                clip_out = clip_model.vision_model(pixel_values=pixel_values)
                patch_features = clip_out.last_hidden_state[:, 1:, :]

            projected = fresh_connector(patch_features)

            bos_emb = model.get_input_embeddings()(bos_id.expand(bs, -1))
            q_embeds = model.get_input_embeddings()(question_ids)
            a_embeds = model.get_input_embeddings()(answer_ids)

            inputs_embeds = torch.cat([bos_emb, projected, q_embeds, a_embeds], dim=1)

            # Labels: mask everything before answer
            prefix_len = 1 + 49 + question_ids.shape[1]
            ignore = torch.full((bs, prefix_len), -100, dtype=torch.long, device=device)
            labels = torch.cat([ignore, answer_ids], dim=1)
            labels[labels == tokenizer.eos_token_id] = -100

            vqa_output = model(inputs_embeds=inputs_embeds, labels=labels)
            vqa_loss = vqa_output.loss

            try:
                alpaca_ids = next(alpaca_iter).to(device)
            except StopIteration:
                alpaca_iter = iter(alpaca_loader)
                alpaca_ids = next(alpaca_iter).to(device)

            alpaca_labels = alpaca_ids.clone()
            alpaca_labels[alpaca_labels == tokenizer.eos_token_id] = -100

            lm_output = model(input_ids=alpaca_ids, labels=alpaca_labels)
            lm_loss = lm_output.loss
            loss = vqa_loss + lambda_val * lm_loss
            loss = loss / accum_steps
            loss.backward()
            vqa_loss_history.append(vqa_loss.item())
            lm_loss_history.append(lm_loss.item())
            combined_loss_history.append(loss.item() * accum_steps)

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_losses.append(loss.item() * accum_steps)

            # This ensures that we update the progress bar with current losses
            pbar.set_postfix({"vqa": f"{vqa_loss.item():.3f}", "lm": f"{lm_loss.item():.3f}"})

            
            if (step + 1) % 300 == 0:
                vqa_acc_quick, ppl_quick = quick_eval(
                    model, fresh_connector, clip_model, vqa_val, full_test,
                    alpaca_subset, processor, tokenizer, device
                )
                # Write to console without breaking the tqdm progress bar
                tqdm.write(f"  [Epoch {epoch+1} | Step {step+1}] VQA acc: {vqa_acc_quick*100:.2f}% | PPL: {ppl_quick:.2f}")

        print(f"Epoch {epoch+1} avg loss: {sum(epoch_losses)/len(epoch_losses):.4f}")

    # Full eval at end
    ppl_after = compute_perplexity(model, alpaca_subset, tokenizer, device)
    R = ppl_after / ppl_0

    # We pass show_progress=True here to get a tqdm bar for the final big validation sweep
    vqa_acc = compute_vqa_accuracy(model, fresh_connector, clip_model, vqa_val, full_test, processor, tokenizer, device, show_progress=True)

    # This ensure weights directory exists
    os.makedirs("weights", exist_ok=True)

    # we save step-by-step losses to an .npz file
    np.savez(f"weights/step_losses_lambda{lambda_val}.npz",
             vqa_loss=vqa_loss_history,
             lm_loss=lm_loss_history,
             combined_loss=combined_loss_history)

    # Save weights
    torch.save(fresh_connector.state_dict(), f"weights/connector_phase2_lambda{lambda_val}.pt")
    model.save_pretrained(f"weights/lora_phase2_lambda{lambda_val}")

    print(f"λ={lambda_val} | VQA acc: {vqa_acc*100:.2f}% | PPL: {ppl_after:.2f} | R: {R:.3f}")

    del model
    del fresh_connector
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return {"lambda": lambda_val, "vqa_acc": vqa_acc * 100, "ppl": ppl_after, "R": R}



def train_various_lambdas(vqa_train, alpaca_subset, vqa_val,
            full_train, full_test, processor, smol_tokenizer,
            clip_model, ppl_0, device, r_norm, epochs=1, batch_size=8, accum_steps=16):
    # Run all 4 lambdas
    results = []
    for lam in [0.0, 0.05, 0.2, 0.5]:
        result = train_phase2(
            lam, vqa_train, alpaca_subset, vqa_val,
            full_train, full_test, processor, smol_tokenizer,
              clip_model, ppl_0, device, r_norm,
            epochs=epochs, batch_size=batch_size, accum_steps=accum_steps
        )
        results.append(result)

    # Print final table
    print("\n\n=== Phase 2 λ Ablation Results ===")
    print(f"{'Condition':<12} {'λ':<6} {'VQA acc (%)':<14} {'R':<6}")
    for cond, res in zip(["No replay", "Weak", "Baseline", "Strong"], results):
        print(f"{cond:<12} {res['lambda']:<6} {res['vqa_acc']:<14.2f} {res['R']:<6.3f}")
    return results