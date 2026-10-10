import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from download_dataset import load_dataset, load_clip_processor
from model_loader import load_model
from connector import Connector
from connector_training import train_connector
from caption_dataset import create_captions_dataset_loader, CaptionDataset
from compute_connector_norm import compute_r_norm
from vqa_inference import vqa_inference
from perplexity_smollm import compute_perplexity
from smol_lm_training import train_various_lambdas
from evaluation_vlm import compute_vqa_accuracy_detailed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_indices, test_indices, captions, cifar_10_classes, vqa_train, vqa_val, full_train, full_test, alpaca_dataset, alpaca_subset = load_dataset()
    clip_processor = load_clip_processor()

    # loading the clip_model, smol_lm, and smol_tokenizer
    smol_lm, smol_tokenizer, clip_model = load_model(full_train, train_indices, clip_processor, clip_eval = False)

    # creating the captions dataset
    _, caption_loader = create_captions_dataset_loader(
        CaptionDataset=CaptionDataset,
        captions=captions,
        full_train=full_train,
        processor=clip_processor,
        smol_tokenizer=smol_tokenizer,
    )

    clip_model.to(device)
    smol_lm.to(device)
    connector = Connector().to(device)
    # training the connector
    #define the optimizer
    connector_optimizer = torch.optim.AdamW(params=connector.parameters(), lr=0.001)
    connector = train_connector(connector, clip_model, smol_lm, smol_tokenizer, caption_loader, connector_optimizer)


    #now lets compute connector r_norm
    r_norm = compute_r_norm(connector, clip_model, smol_lm, caption_loader, device, load_model=False)

    #now lets test inference with the connector trained. 
    connector = vqa_inference(connector, smol_lm, full_test, test_indices, clip_model, r_norm, smol_tokenizer, clip_processor, cifar_10_classes)


    # check smol_lm perplexity before running the code
    ppl_0 = compute_perplexity(smol_lm, smol_tokenizer, alpaca_dataset)


    #now that we have trained connector, we will train the smol_lm
    train_various_lambdas(vqa_train, alpaca_subset, vqa_val,
            full_train, full_test, clip_processor, smol_tokenizer,
            clip_model, ppl_0, device, r_norm,
            epochs=1, batch_size=8, accum_steps=16)

    selected_lambda = 0.2
    base_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-360M-Instruct", torch_dtype=torch.float32
    ).to(device)
    phase3_model = PeftModel.from_pretrained(
        base_model, f"weights/lora_phase2_lambda{selected_lambda}"
    )
    phase3_connector = Connector().to(device)
    phase3_connector.load_state_dict(torch.load(
        f"weights/connector_phase2_lambda{selected_lambda}.pt",
        map_location=device,
    ))
    phase3_connector.scale_factor = 1.0 / r_norm

    compute_vqa_accuracy_detailed(
        phase3_model, phase3_connector, clip_model, vqa_val, full_test,
        clip_processor, smol_tokenizer, device, cifar_10_classes,
        label=f"Phase 2 lambda={selected_lambda}",
    )


if __name__ == "__main__":
    main()
















