
# Analysis of Bias in U.S. History Textbooks Using BERT

This repo provides an example of the gender word prediction study conducted in [Analysis of Bias in U.S. History Textbooks Using BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report069.pdf), using the same fine-tuned BERT model and sample contexts drawn from the same U.S. History Textbook dataset.

In the gender word prediction study, we extract several sample contexts from the textbook dataset, where each context contains a "gender" word and an "interest" word (related to home, work, and achievement) within a 10-token window. Then, we mask out the "gender" word in each context and evaluate BERT’s ability to predict the correct gender in different contexts.

## Setup

Create a python3 virtual environment and install the necessary packages.

```bash
pip install .
```

After cloning the git repo, create a new directory named `finetuned_bert` and download the saved model weights into the directory using the following commands.

```bash
mkdir finetuned_bert
cd finetuned_bert
wget -e robots=off -r --no-host-directories --cut-dirs 4 --no-parent http://infolab.stanford.edu/~paepcke/textbook_bias_model/; rm index*
```

## Usage

The sample contexts have been preprocessed and grouped into three files in the `sample_textbook_contexts` folder (pr represents the normalized probability of predicting the correct gender):
- pr_low.txt (29 examples): pr < 0.25 (high confidence, incorrect gender prediction)
- pr_med.txt (8 examples): 0.45 < pr < 0.55 (low confidence)
- pr_high.txt (37 examples): pr > 0.9999 (high confidence, correct gender prediction)

Each input file is formatted as an array of entries: (`tokens_tensor`, `segments_tensor`, `tokenized_text`, (`gender_index`, `query_index`, `gender_word`, `query_word`)). 
- `tokens_tensor`, `segments_tensor`, `tokenized_text`: input context encoded into the format needed for feeding into BERT
- `gender_index`, `gender_word`: gender word and its index in the input context
- `query_index`, `query_word`: interest word and its index in the input context (where interest word is related to home, work, or achievement)

Run the `predict_mask.py` script to generate predictions for masked gender tokens in the sample contexts provided. Results will be dumped in three files in the `pred_textbook_contexts` folder.

```bash
python predict_mask.py
```

Each output file is formatted as an array of entries: (`tokens_tensor`, `segments_tensor`, `tokenized_text`, (`gender_index`, `query_index`, `gender_word`, `query_word`), `norm_prob`, `top5_preds`). All items in each entry are identical to those in the input file, except for the last two items.
- `norm_prob`: normalized probability of the correct gender prediction (e.g. if the correct gender is female, `norm_prob` would return `pr(female) / (pr(female) + pr(male))`
- `top5_preds`: top 5 predictions for the masked gender token

To view the output files in a nicer format, use the provided notebook: `view_predictions.ipynb`.

## Misc

You can play around with the `feed_sample_sentence()` function in `predict_mask.py` to feed a custom sample input into the fine-tuned BERT model and output the top 5 predictions.