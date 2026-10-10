#load the weights into the connector model
# connector.load_state_dict(torch.load("weights/connector_phaseA1.pt"))
import torch
import tqdm
import numpy as np

def compute_r_norm(connector, clip_model, smol_lm, dataloader, device, load_model = False):
    if load_model:
        connector.load_state_dict(torch.load("weights/connector_phaseA1.pt", map_location="cpu"))
    connector.eval()
    connector_norms = []
    lm_norms = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)
    connector.to(device)
    smol_lm.to(device)
    

    with torch.no_grad():
        for pixel_values, caption_ids in tqdm(dataloader, desc="Computing r_norm"):
            pixel_values = pixel_values.to(device)
            caption_ids = caption_ids.to(device)

            clip_output = clip_model.vision_model(pixel_values=pixel_values)
            patch_features = clip_output.last_hidden_state[:, 1:, :]
            connector_out = connector(patch_features)

            lm_embeds = smol_lm.model.embed_tokens(caption_ids)

            connector_norms.append(connector_out.norm(dim=-1).mean().item())
            lm_norms.append(lm_embeds.norm(dim=-1).mean().item())

    r_norm = np.mean(connector_norms) / np.mean(lm_norms)
    print(f"r_norm: {r_norm:.4f}")
    return r_norm

# # Usage
# r_norm = compute_r_norm(connector, clip_model, smol_lm, caption_loader, device)

# connector.scale_factor = 1.0 / r_norm
# _ = compute_r_norm(connector, clip_model, smol_lm, caption_loader, device)