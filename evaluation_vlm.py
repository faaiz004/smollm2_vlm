from collections import Counter, defaultdict
from tqdm import tqdm
import torch
import gc
from peft import PeftModel
from transformers import AutoModelForCausalLM

def compute_vqa_accuracy_detailed(model, connector, clip_model, vqa_val_data, full_test,
                                   processor, tokenizer, device, cifar_10_classes, label=""):
    model.eval()
    connector.eval()
    bos_id = torch.tensor([[tokenizer.bos_token_id]]).to(device)

    template_correct = defaultdict(int)
    template_total = defaultdict(int)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    overall_correct = 0
    overall_total = 0

    with torch.no_grad():
        for item in tqdm(vqa_val_data, desc=f"Eval {label}"):
            true_label = cifar_10_classes[full_test[item["image_idx"]][1]]
            skill = item["skill"]

            bos_emb = model.get_input_embeddings()(bos_id)
            question_ids = tokenizer(item["question"], return_tensors="pt")["input_ids"].to(device)
            question_embeds = model.get_input_embeddings()(question_ids)

            pil_image = full_test[item["image_idx"]][0]
            pixel_values = processor(images=pil_image, return_tensors="pt")["pixel_values"].to(device)
            clip_out = clip_model.vision_model(pixel_values=pixel_values)
            patch_features = clip_out.last_hidden_state[:, 1:, :]
            projected = connector(patch_features)
            inputs_embeds = torch.cat([bos_emb, projected, question_embeds], dim=1)

            answer_ids = tokenizer(item["answer"], return_tensors="pt")["input_ids"][0].to(device)
            current_embeds = inputs_embeds
            match = True

            for t in range(len(answer_ids)):
                output = model(inputs_embeds=current_embeds)
                predicted_token_id = output.logits[:, -1, :].argmax(dim=-1)

                if predicted_token_id.item() != answer_ids[t].item():
                    match = False
                    break

                next_embed = model.get_input_embeddings()(predicted_token_id.unsqueeze(0))
                current_embeds = torch.cat([current_embeds, next_embed], dim=1)

            overall_correct += int(match)
            overall_total += 1
            template_correct[skill] += int(match)
            template_total[skill] += 1
            class_correct[true_label] += int(match)
            class_total[true_label] += 1

    print(f"\n{'='*60}")
    print(f"  {label} — Exact-Match Accuracy")
    print(f"{'='*60}")
    print(f"\n  Overall: {overall_correct/overall_total*100:.2f}% ({overall_correct}/{overall_total})")

    print(f"\n  {'--- Per Template ---':^40}")
    print(f"  {'Template':<16} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    for skill in sorted(template_total.keys()):
        acc = template_correct[skill] / template_total[skill] * 100
        print(f"  {skill:<16} {acc:>9.2f}% {template_correct[skill]:>10} {template_total[skill]:>8}")

    print(f"\n  {'--- Per Class ---':^40}")
    print(f"  {'Class':<16} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    for cls in sorted(class_total.keys()):
        acc = class_correct[cls] / class_total[cls] * 100
        print(f"  {cls:<16} {acc:>9.2f}% {class_correct[cls]:>10} {class_total[cls]:>8}")

    return {
        "overall": overall_correct / overall_total * 100,
        "per_template": {s: template_correct[s]/template_total[s]*100 for s in template_total},
        "per_class": {c: class_correct[c]/class_total[c]*100 for c in class_total},
    }



# base_model_p3 = AutoModelForCausalLM.from_pretrained(
#     "HuggingFaceTB/SmolLM2-360M-Instruct", torch_dtype=torch.float32
# ).to(device)
# model_p3 = PeftModel.from_pretrained(base_model_p3, "/content/weights/lora_phase3")

# connector_p3 = Connector().to(device)
# connector_p3.load_state_dict(torch.load("/content/weights/connector_phase3.pt"))
# connector_p3.scale_factor = 1.0 / r_norm

# phase3_results = compute_vqa_accuracy_detailed(
#     model_p3, connector_p3, clip_model, vqa_val, full_test,
#     processor, smol_tokenizer, device, label="Phase 3"
# )

# del model_p3, base_model_p3, connector_p3
# gc.collect()
# torch.cuda.empty_cache()