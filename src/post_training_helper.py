import torch
import transformers
import pandas as pd
from bert_layers import BertForMaskedLM
from torch.utils.data import Dataset, DataLoader

# define dataset class with attention mask
class TextDataset(Dataset):
    """Dataset for feeding data to model for incremental training."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):

        super(TextDataset, self).__init__()

        # load data 
        self.df = pd.read_csv(data_path, header=None)
        self.sequences = self.df.iloc[:,0].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i) -> str:
        
        # tokenize data
        tokenized_input = self.tokenizer(self.sequences[i], return_tensors="pt", padding="max_length", max_length=512)#tokenizer.model_max_length)
        
        # extract tokenized input ids
        input_ids = tokenized_input["input_ids"].squeeze(0)
        attention_mask = tokenized_input["attention_mask"].squeeze(0)
        
        return input_ids, attention_mask
    
# define masking function for labels
def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    
    labels_mask = input_ids.clone()
    special_tokens = torch.tensor([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id], device=input_ids.device)
    
    # convert special tokens to 1 and other tokens to 0
    labels_mask[torch.isin(labels_mask, special_tokens)] = 1
    labels_mask[labels_mask != 1] = 0
    
    labels_mask_bool = labels_mask.bool() 
    
    # make prob matrix with the given prob
    prob_mat = torch.full(labels_mask.shape, mlm_probability, device=input_ids.device)
    # convert special tokens to 0
    prob_mat.masked_fill_(labels_mask_bool, value=0.0)
    # pick 15% of non special tokens randomly
    masked_indices = torch.bernoulli(prob_mat).bool()
    
    # convert everything except for the ids found above to -100
    labels = input_ids.clone()
    labels[~masked_indices] = -100
    
    # replace masked tokens with the [MASK] token in the input
    input_ids[masked_indices] = tokenizer.mask_token_id
    
    return input_ids, labels
