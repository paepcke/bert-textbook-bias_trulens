'''
Created on Apr 1, 2022

@author: paepcke

TruLens testing for textbook bias project
'''
from enum import Enum
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
#*****from trulens import visualizations as viz
from trulens.nn.attribution import IntegratedGradients
from trulens.nn.models import get_model_wrapper
from trulens.nn.slices import Cut, OutputCut
from trulens.utils import tru_logger

from predict_mask import feed_sample_sentence, read_context_windows, predict_gender_mask
from nlpviz.nlp_viz import HTMLTable

class RenderStyle(Enum):
    PLAIN = 1
    HTML  = 2

class DisplayContext(Enum):
    TERM = 1
    JUPYTER = 2
    COLAB = 3 

class TruLensTester:
    '''
    Exercise the textbook bias model through TruLens.
    Code modeled on predict_mask.py:main(), which comes
    with the code from the repo 
    '''

    #------------------------------------
    # Constructor
    #-------------------


    def __init__(self):
        '''
        Constructor
        '''
        
        # Directories
        self.pretrained_model_dir = 'bert-base-uncased'
        self.finetuned_model_dir = 'finetuned_bert'
        self.samples_dir = 'sample_textbook_contexts/'
        self.outputs_dir = 'pred_textbook_contexts/'

        self.logger = logging.getLogger(tru_logger.__name__)

        self.analyze_textbook_bias()
        
    #------------------------------------
    # analyze_textbook_bias
    #-------------------
    
    def analyze_textbook_bias(self):
        # Load tokenizer and fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_dir)
        model = AutoModelForMaskedLM.from_pretrained(self.finetuned_model_dir)
        # Put the model in "evaluation" mode, meaning feed-forward operation
        #*******model.eval()
        #*******torch.set_grad_enabled(False)
        
        # Wrap the model:
        
        task = TruLensTextbookWrapper(model, tokenizer)
        task.wrapper = get_model_wrapper(task.model, 
                                         input_shape=(None, task.tokenizer.model_max_length), 
                                         device=task.device,
                                         backend='pytorch'
                                         )

        task.infl_max = IntegratedGradients(
            model = task.wrapper,
            doi_cut=Cut('bert_embeddings_word_embeddings'),
            qoi_cut=OutputCut(accessor=lambda o: o.logits)
            #******qoi_cut=OutputCut(accessor=lambda o: o)
        )
    
        # Feed sample sentences through BERT model
        sample_sentence = 'The doctor bought new treats for ' + tokenizer.mask_token + ' two cats.'
        print(sample_sentence)
        feed_sample_sentence(model, tokenizer, sample_sentence)
        sample_sentence = 'The doctor bought new treats for ' + tokenizer.mask_token + ' two cats and accidentally poisoned them.'
        print(sample_sentence)
        
        # Print the top 5 mask replacement choices:
        word_attributions = self.feed_sample_sentence_trulens(task, sample_sentence)

        # Show the words and their scores:
        html_tbl = HTMLTable(word_attributions)
        html_tbl.render_to_web()
        
        # pr_low.txt: high confidence, incorrect gender prediction
        # pr_med.txt: low confidence
        # pr_high.txt: high confidence, correct gender prediction
        files = ['pr_low.txt', 'pr_med.txt', 'pr_high.txt']
        os.makedirs(self.outputs_dir, exist_ok=True)
    
        # The following runs through all examples. Stop
        # at any time:
        
        print("Will now run through all examples in pr_low.txt et al. Stop when you've seen enough...")
    
        for filename in files:
            dump = []
            data = read_context_windows(os.path.join(self.samples_dir, filename))
            # Run 29 examples through the model without TruLens
            for context in data:
                norm_prob, top5_preds = predict_gender_mask(model, tokenizer, context)
                tokens_tensor, segments_tensor, tokenized_text, mask_info = context
                dump.append({'tokens_tensor'   : tokens_tensor, 
                             'segments_tensor' : segments_tensor, 
                             'tokenized_text'  : tokenized_text,
                             'mask_info'       : mask_info,
                             'norm_prob'       : norm_prob,
                             'top5_preds'      : top5_preds})
                for sentence_tokens in tokens_tensor:
                    sentence = tokenizer.decode(sentence_tokens)
                    word_attributions = self.feed_sample_sentence_trulens(task, sentence)
                    html_tbl.add_rows(word_attributions)


    #------------------------------------
    # feed_sample_sentence_trulens
    #-------------------
    
    def feed_sample_sentence_trulens(self, task, sentence):
        # Encode sentence with a masked token in the middle
        sentence = torch.tensor([task.tokenizer.encode(sentence)])
    
        # Identify the masked token position: 7 in this case:
        masked_index = torch.where(sentence == task.tokenizer.mask_token_id)[1].tolist()[0]
    
        # Get the five top answers, receiving a structure:
        # MaskedLMOutput(loss=None, 
        #                logits=tensor([[[ -7.8824,  -7.9561,  -7.9505,  ...,  -7.6466,  -7.6272,  -5.8483],
        #                                [-14.2105, -13.9069, -13.8993,  ..., -11.5410, -11.3020, -13.8501],
        
        model_result = task.model(sentence)
        top5_indices = model_result.logits[:, masked_index].topk(5).indices
        result = top5_indices.tolist()[0]
        print(task.tokenizer.decode(result))
        
        # Now find the influence of each word using TruLens:

        attrs = task.infl_max.attributions(sentence)
        # We now have sentence:
        #    tensor([[  101,  1996,  3460,  4149,  2047, 18452,  2005,   103,  2048,  8870,
        #              1998,  9554, 17672,  2068,  1012,   102]])
        # sentence.shape: torch.Size([1, 16])
        # And we have attrs: torch.Size([1, 16, 768])
        #
        # The '1' in the shapes indicates that there is only one 
        # sentence involved. The 16 is the number of our single
        # sentence's input tokens. The 768 are embedding dimensions,
        # which we will aggregate into a scalar. 
        #
        # The outer loop runs through each sentence; the inner
        # loop runs through each input token:
        word_attributions = []
        for token_ids, token_attr in zip(sentence, attrs):
            for token_id, token_attr in zip(token_ids, token_attr):
                # Not that each `word_attr` has a magnitude for each of the embedding
                # dimensions, of which there are many. We aggregate them for easier
                # interpretation and display.
                attr = token_attr.sum()
                word = task.tokenizer.decode(token_id)
                word_attributions.append((word, attr))
        
                print(f"{word}({attr:0.3f})", end=' ')
        
            print()

        return word_attributions

# -------------TruLens Wrapper Models -----------------

class TruLensTextbookWrapper:
    MODEL = f"textbook_bias"

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


# ------------------------ Main ------------
if __name__ == '__main__':
    
    TruLensTester()
    
