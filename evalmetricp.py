#!/usr/bin/env python
'''
Evaluation metric for Santa 2024.
This one works in parallel (batch size > 1)
'''

import gc
import os
from math import exp
from collections import Counter
from typing import List, Optional, Union, Dict
import time
import numpy as np
import pandas as pd
import transformers

import torch

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = os.environ['MODEL']

class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    model_path: str = model_path,
    load_in_8bit: bool = True,
    clear_mem: bool = False,
) -> float:
    """
    Calculates the mean perplexity of submitted text permutations compared to an original text.

    Parameters
    ----------
    solution : DataFrame
        DataFrame containing the original text in a column named 'text'.
        Includes a row ID column specified by `row_id_column_name`.

    submission : DataFrame
        DataFrame containing the permuted text in a column named 'text'.
        Must have the same row IDs as the solution.
        Includes a row ID column specified by `row_id_column_name`.

    row_id_column_name : str
        Name of the column containing row IDs.
        Ensures aligned comparison between solution and submission.

    model_path : str
        Path to the serialized LLM.

    clear_mem : bool
        Clear GPU memory after scoring by clearing the CUDA cache.
        Useful for testing.

    Returns
    -------
    float
        The mean perplexity score. Lower is better.

    Raises
    ------
    ParticipantVisibleError
        If the submission format is invalid or submitted strings are not valid permutations.

    Examples
    --------
    >>> import pandas as pd
    >>> model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
    >>> solution = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["this is a normal english sentence", "the quick brown fox jumps over the lazy dog"]
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["sentence english normal a is this", "lazy the over jumps fox brown quick the dog"]
    ... })
    >>> score(solution, submission, 'id', model_path=model_path, clear_mem=True) > 0
    True
    """
    # Check that each submitted string is a permutation of the solution string
    sol_counts = solution.loc[:, 'text'].str.split().apply(Counter)
    sub_counts = submission.loc[:, 'text'].str.split().apply(Counter)
    invalid_mask = sol_counts != sub_counts
    if invalid_mask.any():
        raise ParticipantVisibleError(
            'At least one submitted string is not a valid permutation of the solution string.'
        )

    # Calculate perplexity for the submitted strings
    sub_strings = [
        ' '.join(s.split()) for s in submission['text'].tolist()
    ]  # Split and rejoin to normalize whitespace
    scorer = PerplexityCalculator(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
    )  # Initialize the perplexity calculator with a pre-trained model
    perplexities = scorer.get_perplexity(
        sub_strings
    )  # Calculate perplexity for each submitted string

    if clear_mem:
        # Just move on if it fails. Not essential if we have the score.
        try:
            scorer.clear_gpu_memory()
        except:
            print('GPU memory clearing failed.')

    return float(np.mean(perplexities))


class PerplexityCalculator:
    """
    Calculates perplexity of text using a pre-trained language model.

    Adapted from https://github.com/asahi417/lmppl/blob/main/lmppl/ppl_recurrent_lm.py

    Parameters
    ----------
    model_path : str
        Path to the pre-trained language model

    load_in_8bit : bool, default=True
        Use 8-bit quantization for the model. Requires CUDA.

    device_map : str, default="auto"
        Device mapping for the model.
    """

    def __init__(
        self,
        model_path: str = model_path,
        load_in_8bit: bool = False,
        device_map: str = 'auto',
        device: str = None
            
    ):
        global DEVICE
        if device == 'cpu':
            DEVICE = torch.device('cpu')
            device_map='cpu'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="right")
        # Configure model loading based on quantization setting and device availability
        if load_in_8bit:
            if DEVICE.type != 'cuda':
                raise ValueError('8-bit quantization requires CUDA device')
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
                device_map=device_map,
            )

        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.softmax_fct = torch.nn.LogSoftmax(dim=-1)
        self.model.eval()
        if not load_in_8bit:
            print(f'moving to {DEVICE}')
            self.model.to(DEVICE)  # Explicitly move the model to the device

    def get_perplexity(
        self, input_texts: Union[str, List[str]],
            batch_size = 1,
            ret_logits=False,
    ) -> Union[float, List[float]]:
        """
        Calculates the perplexity of given texts.

        Parameters
        ----------
        input_texts : str or list of str
            A single string or a list of strings.

        batch_size : int, default=None
            Batch size for processing. Defaults to the number of input texts.

        verbose : bool, default=False
            Display progress bar.

        Returns
        -------
        float or list of float
            A single perplexity value if input is a single string,
            or a list of perplexity values if input is a list of strings.

        Examples
        --------
        >>> import pandas as pd
        >>> model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
        >>> scorer = PerplexityCalculator(model_path=model_path)

        >>> submission = pd.DataFrame({
        ...     'id': [0, 1, 2],
        ...     'text': ["this is a normal english sentence", "thsi is a slihgtly misspelled zr4g sentense", "the quick brown fox jumps over the lazy dog"]
        ... })
        >>> perplexities = scorer.get_perplexity(submission["text"].tolist())
        >>> perplexities[0] < perplexities[1]
        True
        >>> perplexities[2] < perplexities[0]
        True

        >>> perplexities = scorer.get_perplexity(["this is a sentence", "another sentence"])
        >>> all(p > 0 for p in perplexities)
        True

        >>> scorer.clear_gpu_memory()
        """
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []
        start_index = 0
        num = batch_size
        #add special tokens to everyone
        texts_with_special = [self.tokenizer.bos_token + text + self.tokenizer.eos_token for text in input_texts]
        start_time = time.time()
        ret_logits_value = []
        while start_index < len(input_texts):
            with torch.no_grad():
                if len(input_texts) < num + start_index:
                    num = len(input_texts)
                # Tokenize
                #print(f'DEBUG {start_index=} {num=}')
                model_inputs = self.tokenizer(
                    texts_with_special[start_index:start_index+num],
                    return_tensors='pt',
                    add_special_tokens=False,
                    padding=True
                )
                #print(f'{model_inputs=}')
                if 'token_type_ids' in model_inputs:
                    model_inputs.pop('token_type_ids')
                PRINT_TOKENS=False
                if PRINT_TOKENS:
                    for a in model_inputs['input_ids']:
                        print(f'one')
                        for tid in a:
                            s=self.tokenizer.decode(tid, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
                            print(f'{tid=} /{s}/')

                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

                # Get model output
                output = self.model(**model_inputs, use_cache=False)
                #print(f'{output=}')
                logits = output['logits']
                if ret_logits:
                    ret_logits_value.append(logits.detach().cpu().numpy())
                label = model_inputs["input_ids"]
                #print(f'label {label}')
                #print(f'logits {logits}')
                #print(f'pad token id {self.tokenizer.pad_token_id=}')
                #print(f'{PAD_TOKEN_LABEL_ID=}')
                label[label == self.tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

                # Shift logits and labels for calculating loss
                shift_logits = logits[..., :-1, :].contiguous()  # Drop last prediction
                shift_labels = label[..., 1:].contiguous()  # Drop first input
                #print(f'{shift_logits.size()=}')
                # Calculate token-wise loss
                #NOTE: worry about PAD TOKENS - see version 1 of Metric Notebook
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = loss.view(len(logits), -1)
                #print(loss)
                #print(loss.dtype, loss.type())
                #print(f'get_perp {loss=}')
                #print(f'get_perp {label=}')
                valid_length = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
                loss = torch.sum(loss, -1) / valid_length
                loss_list += loss.detach().cpu().tolist()
                start_index += num
                del output
                del logits

        ppl = [exp(i) for i in loss_list]

        if ret_logits:
            #log = [x.detach().cpu().numpy() for x in ret_logits_value]
            #for x in ret_logits_value:
            #    print(f'shape of logits before stack {x.shape}')
            #now stack into big array
            #need concatenate if already have the axis
            #print(f'{ret_logits_value[0].shape=}')
            if len(ret_logits_value)==1:
                ret = ret_logits_value[0]
            elif len(ret_logits_value[0])==3:
                ret = np.concatenate(ret_logits_value, axis=0)
            else:
                ret = np.stack(ret_logits_value)
            #print(f'after stack {ret.shape}')
            return ppl[0] if single_input else ppl, ret
        else:
            return ppl[0] if single_input else ppl

    def get_perplexity2(
        self, input_texts: Union[str, List[str]],
            batch_size=1,
            ret_logits=False,
            ret_losses=False,
            ret_logsoftmax=False,  
            verbose=False
    ) -> Dict[str, Union[float, List[float]]]:
        #returns dict ['perp'] = perplexities as get_perplexity (list),
        #             ['logits'] = logits, an numpy array
        #             ['losses'] = total loss as a list for each text, with total loss
        #                      up to each word. e.g. [2, 4, 6, 10] where
        #                      the last entry is the total loss (= log(perplexity))
        #             ['valid_tokens'] = list of number of a valid tokens
        #             ['logsoftmax'] = log_softmax of logits as a numpy array
        #Let's just make sure we always get a list, to simplify our dimensions later
        assert isinstance(input_texts, list) 
        #print(f'inputs  {input_texts}')
        loss_list = []
        #total_loss_list = []
        start_index = 0
        num = batch_size
        #add special tokens to everyone and convert to a list of words instead of a string
        #need to add spaces to all but first word
        space=lambda x: ' ' if x>0 else ''
        texts_with_special = [[self.tokenizer.bos_token] + [space(i) + w for i,w in enumerate(text.split())] + [self.tokenizer.eos_token] for text in input_texts]
        start_time = time.time()
        ret_logits_value = []
        ret_softmax_value = []
        accum_loss=[]
        valid_tokens_list = []
        while start_index < len(input_texts):
            with torch.no_grad():
                if len(input_texts) < num + start_index:
                    num = len(input_texts)
                model_inputs = self.tokenizer(
                    texts_with_special[start_index:start_index+num],
                    return_tensors='pt',
                    add_special_tokens=False,
                    padding=True,
                    is_split_into_words=True,
                )
                #print(f'{model_inputs=}')
                if 'token_type_ids' in model_inputs:
                    model_inputs.pop('token_type_ids')
                #print(f'{model_inputs["input_ids"]=}')
                PRINT_TOKENS=verbose
                if PRINT_TOKENS:
                    for a in model_inputs['input_ids']:
                        print(f'one')
                        for tid in a:
                            s=self.tokenizer.decode(tid, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
                            print(f'tid {tid.item():10}   /{s}/')
                ends = {}  #sort ends[b][word_index]=end index for loss
                #for b in range(num):
                #    print(f'words b{b} {model_inputs.word_ids(b)}')
                #NOTE: word_ids and word_to_tokens only works with is_split_into_words=True in tokenizer call
                #print(model_inputs['input_ids'])
                #We need to be careful because of batching
                for b in range(num):
                    #print('Z', b, model_inputs.word_ids(b))
                    ends[b]={}
                    for w_idx in set(model_inputs.word_ids(b)):
                        if w_idx is None:
                            #this is padding, no more words
                            continue
                        start, end = model_inputs.word_to_tokens(batch_or_word_index=b,word_index=w_idx)
                        #print(f'batch {b} {w_idx} {start=} {end=}')
                        #record the end token position for this word
                        ends[b][w_idx]=end
                #Move inputs to GPU.  word_ids will no longer be available
                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
                # Get model output
                output = self.model(**model_inputs, use_cache=False)
                #print(f'{output=}')
                logits = output['logits']
                if ret_logits:
                    ret_logits_value.append(logits.detach().cpu().numpy())
                if ret_logsoftmax:
                    #print(f'{logits.shape=}')
                    ret_softmax_value.append(self.softmax_fct(logits).detach().cpu().numpy())
                label = model_inputs["input_ids"]
                #print(f'label {label.shape} {label}')
                #print(f'logits {logits}')
                #print(f'pad token id {self.tokenizer.pad_token_id=}')
                #print(f'{PAD_TOKEN_LABEL_ID=}')
                label[label == self.tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

                # Shift logits and labels for calculating loss
                shift_logits = logits[..., :-1, :].contiguous()  # Drop last prediction
                shift_labels = label[..., 1:].contiguous()  # Drop first input
                #print(f'{shift_logits.size()=}')
                #print(f'{shift_labels.size()=}')
                #print(f'{shift_labels=}')
                # Calculate token-wise loss
                #NOTE: worry about PAD TOKENS - see version 1 of Metric Notebook
                valid_length = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
                #print(f'{valid_length=}')
                #print(f'{shift_labels.size()=}')
                #print(f'{shift_labels=}')

                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = loss.view(len(logits), -1)
                #print(f'{loss.shape=}')
                #print(f'{shift_labels=}')
                #print(f'get_perp2 {loss=}')
                #print(f'get_perp2 {label=}')
                #Sum the loss from beginning to each word
                for b in range(num):
                    loss_this_batch=[]
                    for windex in range(1, len(texts_with_special[start_index+b])):
                        #print(f'{b=} {windex=} {ends[b][windex-1]=}')
                        z = sum(loss[b,:ends[b][windex-1]])
                        #print(f'{loss[b, 2]=}')
                        loss_this_batch.append(z.detach().cpu().item())
                    accum_loss.append(loss_this_batch)
                    valid_tokens_list.append( valid_length[b].detach().cpu().item())
                #print(f'{accum_loss=}')
                #for b in range(num):
                #    print(f'exp loss {[exp(a/valid_length[b]) for a in accum_loss[b]]}')
                tot_loss = torch.sum(loss, -1)
                #print(f'{valid_length=}')   #tensor([8,8])
                #print(f'tot_loss {tot_loss.shape}')
                loss = tot_loss / valid_length
                #iterate o
                # Calculate average loss
                #print(f'{loss=}')
                #sequence_loss = loss.sum() / len(loss)
                loss_list += loss.detach().cpu().tolist()
                #total_loss_list += tot_loss.detach().cpu().tolist()
                start_index += num
                del output
                del logits
        ret = {}
        #loss_list is a list of the losses for each text
        #accum_loss is a list of [loss(1), loss(2),... loss(lastword)]
        ppl = [exp(i) for i in loss_list]
        ret['perp'] = ppl
        if ret_losses:
            #losses are the accumulated losses after each word
            #[ [word1, word2], [word1, word2] ...]
            ret['losses'] = accum_loss
            #[num_tokens in text1, num_tokens in text2, ...]
            ret['valid_tokens']=valid_tokens_list
            ret['ends']=ends
        if ret_logits:
            #log = [x.detach().cpu().numpy() for x in ret_logits_value]
            #for x in ret_logits_value:
            #    print(f'shape of logits before stack {x.shape}')
            #now stack into big array
            #need concatenate if already have the axis
            #print(f'{len(ret_logits_value)=}')
            #print(f'{ret_logits_value[0].shape=}')
            if len(ret_logits_value)==1:
                logits_to_ret = ret_logits_value[0]
            elif len(ret_logits_value[0])==3:
                logits_to_ret = np.concatenate(ret_logits_value, axis=0)
            else:
                logits_to_ret = np.stack(ret_logits_value)
            #print(f'after stack {logits_to_ret.shape}')
            ret['logits']=logits_to_ret
        if ret_logsoftmax:
            #log = [x.detach().cpu().numpy() for x in ret_logits_value]
            #for x in ret_logits_value:
            #    print(f'shape of logits before stack {x.shape}')
            #now stack into big array
            #need concatenate if already have the axis
            #print(f'{len(ret_logits_value)=}')
            #print(f'{ret_logits_value[0].shape=}')
            if len(ret_softmax_value)==1:
                softmax_to_ret = ret_softmax_value[0]
            elif len(ret_softmax_value[0])==3:
                softmax_to_ret = np.concatenate(ret_softmax_value, axis=0)
            else:
                softmax_to_ret = np.stack(ret_softmax_value)
            #print(f'after stack {logits_to_ret.shape}')
            ret['logsoftmax']=softmax_to_ret
        return ret
    
    def clear_gpu_memory(self, tokenizer=False, model=False) -> None:
        """Clears GPU memory by deleting references and emptying caches."""
        if not torch.cuda.is_available():
            return

        # Delete model and tokenizer if they exist
        if model and hasattr(self, 'model'):
            del self.model
        if tokenizer and hasattr(self, 'tokenizer'):
            del self.tokenizer

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache and reset memory stats
        with DEVICE:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    #test on sample_submission.csv
    
    number_to_do = 1
    batch_size = 1
    df = pd.read_csv('data/sample_submission.csv')
    scorer = PerplexityCalculator(model_path, load_in_8bit=False, device='cpu')
    start = time.time()
    all_lists = []
    for idx, text in enumerate(df["text"].to_list()):
        words = text.split()
        input_text = " ".join(words)
        if idx==5:
            batch_size=16
        perps = scorer.get_perplexity2([input_text] * batch_size,
                                       batch_size=batch_size,
                                       ret_logits=True,
                                       ret_losses=True)
        print(perps)
        print(f'duration {time.time()-start}')
