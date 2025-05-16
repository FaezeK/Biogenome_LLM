import torch
import transformers
import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import evaluate
import timeit

# timing the run-time
start_time = timeit.default_timer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = transformers.AutoTokenizer.from_pretrained("saved_models", trust_remote_code=True)
mlm_model = transformers.AutoModelForMaskedLM.from_pretrained("saved_models", trust_remote_code=True)

# define dataset class with attention mask
class TextDataset(Dataset):
    """Dataset for evaluating model performance."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):

        super(TextDataset, self).__init__()

        # load data 
        self.df = pd.read_csv(data_path, header=None)
        self.sequences = self.df.iloc[:,0].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        
        tokenized_input = self.tokenizer(self.sequences[i], return_tensors="pt", padding="max_length", max_length=512)#tokenizer.model_max_length)
        
        input_ids = tokenized_input["input_ids"].squeeze(0)
        attention_mask = tokenized_input["attention_mask"].squeeze(0)
        
        return input_ids, attention_mask
        

### Test model performance on chr I of c elegans
data_path = '/data/c_elegans'
test_dataset = TextDataset(tokenizer=tokenizer, data_path=os.path.join(data_path, sys.argv[1]))

mlm_model.to(device)
mlm_model.eval()

### Evaluate the model
pred_df, val_cnt_df, perplexity = evaluate.eval_dnabert2(test_dataset, mlm_model, tokenizer)

pred_df.to_csv(sys.argv[2]+'_'+sys.argv[3]+'_pred_'+sys.argv[4]+'.txt', sep='\t', index=False, header=False)
val_cnt_df.to_csv(sys.argv[2]+'_'+sys.argv[3]+'_top_tokens_'+sys.argv[4]+'.txt', sep='\t', index=False)
pd.Series(perplexity).to_csv(sys.argv[2]+'_'+sys.argv[3]+'_perplexity_'+sys.argv[4]+'.txt', sep='\t', index=False, header=False)

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
