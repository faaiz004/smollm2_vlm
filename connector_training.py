import torch
import torch.nn as nn
from tqdm import tqdm
# Save connector weights
import os


def train_connector(connector: nn.Module, clip_model, smol_lm, smol_tokenizer, caption_loader, optimizer):
    # Training loop
    connector.train()
    epoch_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    connector.to(device)
    clip_model.to(device)
    smol_lm.to(device)

    pbar = tqdm(caption_loader, desc="Phase 1")
    for pixel_values, caption_ids in pbar:
        pixel_values = pixel_values.to(device)
        caption_ids = caption_ids.to(device)
        batch_size = pixel_values.shape[0]

        with torch.no_grad():
            # Get CLIP features — drop CLS token
            clip_output = clip_model.vision_model(pixel_values=pixel_values)
            patch_features = clip_output.last_hidden_state[:, 1:, :]

            # Get BOS embedding and caption embeddings
            bos_id = torch.tensor([[smol_tokenizer.bos_token_id]]).to(device)
            bos_emb = smol_lm.model.embed_tokens(bos_id.expand(batch_size, -1)) # stretch the first dimension to batch size
            caption_embeds = smol_lm.model.embed_tokens(caption_ids)

        # Project visual features through connector (this is what we're training)
        projected = connector(patch_features)

        # Build sequence: [BOS, V1:49, caption_embs]
        inputs_embeds = torch.cat([bos_emb, projected, caption_embeds], dim=1)

        # Build labels: [-100] * (1 + 49) + caption_ids
        # this is basically masking the first 50 position for the loss computation
        ignore_labels = torch.full((batch_size, 50), -100, dtype=torch.long, device=device)
        labels = torch.cat([ignore_labels, caption_ids], dim=1)
        labels[labels == smol_tokenizer.eos_token_id] = -100  # Mask padding tokens

        # Forward pass
        output = smol_lm(inputs_embeds=inputs_embeds, labels=labels)
        loss = output.loss

        # Backward pass — only updates connector
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"Phase 1 avg loss: {sum(epoch_losses) / len(epoch_losses):.4f}")
    os.makedirs("weights", exist_ok=True)   
    torch.save(connector.state_dict(), "weights/connector_phaseA1.pt")
    return connector


