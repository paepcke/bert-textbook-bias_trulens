import os
import ast

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Directories
pretrained_model_dir = 'bert-base-uncased'
finetuned_model_dir = 'finetuned_bert'
samples_dir = 'sample_textbook_contexts/'
outputs_dir = 'pred_textbook_contexts/'


def feed_sample_sentence(model, tokenizer, sentence):
    """Feed a sample sentence with a [MASK] token through the fine-tuned BERT model.
    Print the top 5 predictions for the masked token.
    """
    # Encode sentence with a masked token in the middle
    sentence = torch.tensor([tokenizer.encode(sentence)])

    # Identify the masked token position
    masked_index = torch.where(sentence == tokenizer.mask_token_id)[1].tolist()[0]

    # Get the five top answers
    result = model(sentence)
    result = result[0][:, masked_index].topk(5).indices
    result = result.tolist()[0]
    print(tokenizer.decode(result))


def read_context_windows(path):
    """Read in data from path, where path contains an array of entries:
        (tokens_tensor, segments_tensor, tokenized_text,
        (gender_index, query_index, gender_word, query_word))
    """
    with open(path) as f:
        data = f.read()
        data = data.replace('tensor', '')
        data = ast.literal_eval(data)
    return data


def _prepare_masked_input(tokenizer, context):
    """Take an input context tuple:
        (tokens_tensor, segments_tensor, tokenized_text,
        (gender_index, query_index, gender_word, query_word))
    Prepare the masked input for feeding into BERT.
    """
    mask_token, mask_id = tokenizer.mask_token, tokenizer.mask_token_id
    tokens_tensor, segments_tensor, tokenized_text, mask_info = context
    tokens_tensor = torch.tensor(tokens_tensor)
    segments_tensor = torch.tensor(segments_tensor)

    # Replace gender token with [MASK] token
    gender_index, query_index, gender_word, query_word = mask_info
    tokenized_text[gender_index] = mask_token
    tokens_tensor[0][gender_index] = mask_id
    return tokens_tensor, tokenized_text

def _get_mask_probabilities(model, tokenizer, inputs, masked_position):
    """Feed the input through BERT and return:
    1. Probability distribution of predicted words for [MASK] token.
    2. Top 5 predicted words for [MASK] token.
    """
    # Feed through BERT
    outputs = model(inputs)
    last_hidden_state = outputs[0].squeeze(0)
    # Only get output for the masked token (output is the size of the vocabulary)
    mask_hidden_state = last_hidden_state[masked_position]
    # Convert to probabilities (softmax), giving a probability for each item in the vocabulary
    softmax = torch.nn.Softmax(dim=0)
    probs = softmax(mask_hidden_state)
    # Get top 5 predictions
    top5_preds = outputs[0][:, masked_position].topk(5).indices.tolist()[0]
    top5_preds = tokenizer.decode(top5_preds)
    return probs, top5_preds

def predict_gender_mask(model, tokenizer, context):
    """Feed context into BERT and get prediction for the masked gender token.
    Aggregate total probability for each gender and compute the normalized probability for the correct gender prediction.
    """
    # Parse example context into input tensors and mask the gender word
    _, _, _, sentence_info = context
    gender_index, _, gender_word, _ = sentence_info
    tokens_tensor, tokenized_text = _prepare_masked_input(tokenizer, context)
    probs, top5_preds = _get_mask_probabilities(model, tokenizer, tokens_tensor, gender_index)

    # Compute the normalized probability of the correct gender prediction
    man_words = set(['man', 'men', 'male', 'he', 'him', 'his'])
    woman_words = set(['woman', 'women', 'female', 'she', 'her', 'hers'])

    man_prob = 0
    woman_prob = 0
    for m_word in man_words:
        pronoun_id = tokenizer.convert_tokens_to_ids(m_word)
        man_prob += probs[pronoun_id].item()
    for w_word in woman_words:
        pronoun_id = tokenizer.convert_tokens_to_ids(w_word)
        woman_prob += probs[pronoun_id].item()
    gender_prob = man_prob if gender_word in man_words else woman_prob
    opp_gender_prob = woman_prob if gender_word in man_words else man_prob
    norm_prob = gender_prob / (gender_prob + opp_gender_prob)
    return norm_prob, top5_preds


def main():
    # Load tokenizer and fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
    model = AutoModelForMaskedLM.from_pretrained(finetuned_model_dir)

    # Put the model in "evaluation" mode, meaning feed-forward operation
    model.eval()
    torch.set_grad_enabled(False)

    # Feed sample sentences through BERT model
    sample_sentence = 'The doctor bought new treats for ' + tokenizer.mask_token + ' two cats.'
    print(sample_sentence)
    feed_sample_sentence(model, tokenizer, sample_sentence)
    sample_sentence = 'The doctor bought new treats for ' + tokenizer.mask_token + ' two cats and accidentally poisoned them.'
    print(sample_sentence)
    feed_sample_sentence(model, tokenizer, sample_sentence)

    # pr_low.txt: high confidence, incorrect gender prediction
    # pr_med.txt: low confidence
    # pr_high.txt: high confidence, correct gender prediction
    files = ['pr_low.txt', 'pr_med.txt', 'pr_high.txt']
    os.makedirs(outputs_dir, exist_ok=True)

    for filename in files:
        dump = []
        data = read_context_windows(samples_dir + filename)
        for context in data:
            norm_prob, top5_preds = predict_gender_mask(model, tokenizer, context)
            tokens_tensor, segments_tensor, tokenized_text, mask_info = context
            dump.append((tokens_tensor, segments_tensor, tokenized_text, mask_info, norm_prob, top5_preds))
        
        # dump analysis to outputs dir
        with open(outputs_dir + filename, "w") as output:
            output.write(str(dump))

if __name__ == '__main__':
    main()
