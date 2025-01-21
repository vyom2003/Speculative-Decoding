import torch
from transformers import AutoTokenizer, T5Model
from multiprocessing import Pool
from nltk.translate.bleu_score import sentence_bleu
import time

def sample(logits, temperature=0):
    if temperature == 0:
        return torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    else:
        scaled_logits = logits[:, -1, :] / temperature
        scaled_probabilities = torch.softmax(scaled_logits, dim=-1)
        next_token_id = torch.multinomial(scaled_probabilities, num_samples=1)
        return next_token_id

def calculate_bleu(reference, prediction):
    return sentence_bleu([reference], prediction[2:-2])

def baseline_decode(input_text, model_name="google-t5/t5-large",temp=0):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5Model.from_pretrained(model_name)
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=False).input_ids
    prefix = torch.tensor([[model.config.pad_token_id]])
    start_time = time.time()
    num_ops = 0
    while prefix[0][-1].item() != model.config.eos_token_id:
        outputs = model(input_ids=input_ids, decoder_input_ids=prefix)
        logits = outputs.last_hidden_state @ model.shared.weight.T
        next_token_id = sample(logits, temp)
        prefix = torch.cat([prefix, torch.tensor(next_token_id)], dim=1)
        num_ops+=1
        print(tokenizer.decode(prefix[0].tolist(), skip_special_tokens=True))
    time_taken = time.time() - start_time
    del model
    return tokenizer.decode(prefix[0].tolist(), skip_special_tokens=True), time_taken, num_ops

