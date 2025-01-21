import torch
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import math

def sample(logits, temperature=0):
    if temperature == 0:
        return torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    else:
        scaled_logits = logits[:, -1, :] / temperature
        scaled_probabilities = torch.softmax(scaled_logits, dim=-1)
        next_token_id = torch.multinomial(scaled_probabilities, num_samples=1)
        return next_token_id

def gamma_tokens(model, prefix, input_ids, temperature=0, gamma=3):
    generated_logits, generated_tokens = [], []
    for i in range(gamma):
        outputs = model(input_ids=input_ids, decoder_input_ids=prefix)
        logits = outputs.logits
        generated_logits.append(torch.softmax(logits, dim=-1))
        next_token_id = sample(logits, temperature)
        generated_tokens.append(next_token_id)
        prefix = torch.cat([prefix, next_token_id], dim=-1)
    return generated_logits, generated_tokens

def generate_token(model, input_ids, prefix, temp):
    outputs = model(input_ids=input_ids, decoder_input_ids=prefix)
    logits = outputs.logits
    return torch.softmax(logits.detach(), dim=-1)

def calculate_optimal_gamma(alpha, cost):
    def f(x, alp, c):
        return (1 - alp**(x + 1)) / ((1 - alp) * (x * c + 1))

    def f_prime(x, alp, c):
        term1 = -np.log(alp) * alp**(x + 1) * (1 - alp) * (x * c + 1)
        term2 = (1 - alp**(x + 1)) * (1 - alp) * c
        denominator = ((1 - alp) * (x * c + 1))**2
        return (term1 - term2) / denominator

    def gradient_ascent_exponential(alp, c, u0=0.0, eta=0.1, tol=1e-6, max_iter=1000):
        u = u0 
        for _ in range(max_iter):
            x = np.exp(u)
            grad_x = f_prime(x, alp, c) * x
            if abs(grad_x) < tol:
                break
            u += eta * grad_x
        x_max = np.exp(u)
        return x_max, f(x_max, alp, c)
    
    alpha = min(0.9, max(0.1, alpha))
    gamma, _ = gradient_ascent_exponential(alpha, cost)
    return max(math.ceil(gamma), 1)

def calculate_bleu(reference, prediction):
    return sentence_bleu([reference], prediction[2:-2])