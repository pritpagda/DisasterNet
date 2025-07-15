from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def extract_attention_weights(bert_model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attentions = outputs.attentions  # list of (batch, heads, seq_len, seq_len)

    last_layer = attentions[-1][0]  # first batch, shape (heads, seq_len, seq_len)
    avg_attention = last_layer.mean(dim=0)  # average heads, shape (seq_len, seq_len)
    cls_attention = avg_attention[0]  # attention from CLS token to others

    scores = cls_attention.cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_weights = [{"word": token, "weight": float(score)} for token, score in zip(tokens, scores)]

    return token_weights




