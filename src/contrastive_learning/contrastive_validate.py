from sacrebleu import corpus_bleu
import torch

def validate(model, dataloader, tokenizer):
    model.eval()
    all_hyps = []
    all_refs = []
    with torch.no_grad():
        for batch in dataloader:
            # Prepare inputs
            src_input_ids = batch['src_input_ids'].to(device)
            src_attention_mask = batch['src_attention_mask'].to(device)
            # Generate predictions
            generated_ids = model.generate(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask,
                max_length=tokenizer.model_max_length,
                num_beams=5
            )
            # Decode predictions
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # References
            refs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['contrast_target_input_ids']]
            all_hyps.extend(preds)
            all_refs.extend(refs)

    # Compute BLEU score
    bleu = corpus_bleu(all_hyps, [all_refs])
    print(f"BLEU score: {bleu.score}")

    model.train()
