import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def vqa_inference(connector, smol_lm, full_test, test_indices, clip_model, r_norm, smol_tokenizer, clip_processor, cifar_10_classes):

    connector.eval()
    smol_lm.eval()

    configs = [
        (False, False, "WITHOUT Rescaling | Greedy | Visual tokens only"),
        (True, False, "WITH Rescaling | Greedy | Visual tokens only"),
        (False, True, "WITHOUT Rescaling | Greedy | With prompt"),
        (True, True, "WITH Rescaling | Greedy | With prompt"),
    ]

    prompt = "What is the class in this image?"

    for use_scale, use_prompt, title in configs:

        if use_scale:
            connector.scale_factor = 1.0 / r_norm
        else:
            connector.scale_factor = 1.0

        print(f"\n{'='*60}")
        print(title)
        print(f"{'='*60}")

        for i in range(5):
            image = full_test[test_indices[i]][0]
            label = full_test[test_indices[i]][1]

            pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

            with torch.no_grad():
                clip_output = clip_model.vision_model(pixel_values=pixel_values)
                patch_features = clip_output.last_hidden_state[:, 1:, :]
                projected = connector(patch_features)
                bos_emb = smol_lm.model.embed_tokens(torch.tensor([[smol_tokenizer.bos_token_id]]).to(device))

                if use_prompt:
                    prompt_ids = smol_tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                    prompt_embeds = smol_lm.model.embed_tokens(prompt_ids)
                    inputs_embeds = torch.cat([bos_emb, projected, prompt_embeds], dim=1)
                else:
                    inputs_embeds = torch.cat([bos_emb, projected], dim=1)

                generated_ids = []
                for _ in range(20):
                    output = smol_lm(inputs_embeds=inputs_embeds)
                    next_token_id = output.logits[:, -1, :].argmax(dim=-1)

                    if next_token_id.item() == smol_tokenizer.eos_token_id:
                        break

                    generated_ids.append(next_token_id.item())
                    next_embed = smol_lm.model.embed_tokens(next_token_id.unsqueeze(0))
                    inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

                generated_text = smol_tokenizer.decode(generated_ids)

            print(f"  True: {cifar_10_classes[label]:12s} | Gen: {generated_text}")

    connector.scale_factor = 1.0 / r_norm
    return connector