import torch


def extract_attention_weights(bert_model, input_ids, attention_mask, tokenizer):
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attentions = outputs.attentions

    last_layer = attentions[-1][0]
    avg_attention = last_layer.mean(dim=0)
    cls_attention = avg_attention[0]

    scores = cls_attention.cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_weights = [{"word": token, "weight": float(score)} for token, score in zip(tokens, scores)]

    return token_weights
