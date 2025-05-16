import torch
import sys
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# define function to predict masked tokens in sequences
def eval_dnabert2(test_dataset, model, tokenizer):
    seq_position = 1
    # lists to store token positions
    start_position_ls = []
    end_position_ls = []
    # list to store prediction score (1 or 0)
    score_ls = []
    # list to store all predicted tokens and all prediction probabilities
    all_predicted_tokens = []
    all_pred_probs = []
    # list to store log prob to find perplexity
    all_log_probs = []
    
    # set up data loader
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for batch_idx, batch in enumerate(data_loader):
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        
        sequence_len = input_ids.size(1)
        special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
        
        predicted_token_ids = []
        predicted_token_probs = []
        indices = [] # indices of non-special tokens
        
        # masking tokens once per repeat
        for token_idx in range(sequence_len):
            if input_ids[0, token_idx].item() not in special_tokens:
                masked_input_ids = input_ids.clone()
                masked_input_ids[0, token_idx] = tokenizer.mask_token_id
        
                with torch.no_grad():
                    mlm_res = mlm_model(input_ids=masked_input_ids, attention_mask=attention_mask)
                
                logits = mlm_res.logits[0, token_idx, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_token_id = torch.argmax(probs).item()
                predicted_token_ids.append(predicted_token_id)
                # get prediction prob
                predicted_token_prob = np.round(torch.max(probs).item(), 2)
                predicted_token_probs.append(predicted_token_prob)
                
                # to calculate perplexity, find log prob
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                correct_log_probs = log_probs[input_ids[0, token_idx]].item()
                all_log_probs.append(correct_log_probs)
                
                # get token ids of non-special tokens
                indices.append(token_idx)
        
        # remove special tokens
        true_tokens_input_ids = input_ids.squeeze()[indices]
        
        # get tokens to extract their length in the for loop
        tokens = tokenizer.convert_ids_to_tokens(true_tokens_input_ids)
        
        # get all predicted token ids to find the most frequent predicted tokens later
        all_predicted_tokens.extend(predicted_token_ids)
        all_pred_probs.extend(predicted_token_probs)
        
        # add 1 to pred list if the prediction is correct, add 0 otherwise
        for i in range(true_tokens_input_ids.size()[0]):
            if predicted_token_ids[i] == true_tokens_input_ids[i].item():
                pred_val = 1
            else:
                pred_val = 0
                
            # find the start and end positions of tokens
            start_pos = seq_position
            seq_position += len(tokens[i])
            end_pos = seq_position - 1
            
            # add the positions and predictions to relevant lists
            start_position_ls.append(start_pos)
            end_position_ls.append(end_pos)
            score_ls.append(pred_val)
            
    # make the chr column
    chrom = [str(sys.argv[2])] * len(start_position_ls)
    pred_df = pd.DataFrame({'chrom':chrom, 'start':start_position_ls, 'end':end_position_ls, 'pred':score_ls, 'pred_prob':all_pred_probs})
    
    # get the most predicted tokens
    val_cnt = pd.Series(all_predicted_tokens).value_counts()
    top_frequently_predicted_tokens = tokenizer.convert_ids_to_tokens(list(val_cnt.index[0:3]))
    val_cnt_df = pd.DataFrame({'token':top_frequently_predicted_tokens, 'freq':val_cnt[0:3]})
    
    # Compute perplexity
    avg_log_prob = torch.as_tensor(all_log_probs).mean()
    perplexity = torch.exp(-avg_log_prob).item()
        
    return pred_df, val_cnt_df, perplexity
