import torch
import transformers
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import post_training_helper as pth
from bert_layers import BertForMaskedLM
from torch.utils.data import Dataset, DataLoader

# timing the run-time
start_time = timeit.default_timer()

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and tokenizer
model = BertForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M")
tokenizer = transformers.AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!", flush=True)
    model = torch.nn.DataParallel(model)

model.to(device)

# incremental training
# read data and perform tokenization
data_path = '/data'
file_name_train = 'train.csv'
file_name_valid = 'valid.csv'

train_dataset = pth.TextDataset(tokenizer=tokenizer, data_path=os.path.join(data_path, file_name_train))
valid_dataset = pth.TextDataset(tokenizer=tokenizer, data_path=os.path.join(data_path, file_name_valid))

# set up data loader
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)#, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=48, shuffle=True)#, num_workers=8)

# set up optimizer
lr = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# incremental training for n epochs
epochs = 25
# number of training steps
num_training_steps = len(train_loader) * epochs  
num_warmup_steps = int(0.2 * num_training_steps) 
best_val_loss = float("inf")

# define scheduler with 10% warm up steps and linear decay
scheduler = transformers.get_scheduler(name="linear", optimizer=optimizer, 
                                       num_warmup_steps=num_warmup_steps, 
                                       num_training_steps=num_training_steps)

# record all loss values per epoch
avg_train_loss_ls = []
avg_val_loss_ls = []

# record loss as training goes on
f_1 = open('loss_vals.txt', 'w')

for epoch in range(epochs):
    # training phase
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        
        # mask 15% of the label tokens for prediction
        input_ids, labels = pth.mask_tokens(input_ids, tokenizer, mlm_probability=0.15)
        
        #input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if batch_idx % 1000 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}", flush=True)
            
        train_loss += loss.item()
        num_batches += 1
        
    avg_train_loss = train_loss / num_batches
    print(f"Epoch {epoch + 1}/{epochs} completed. Loss: {avg_train_loss}", flush=True)
    avg_train_loss_ls.append(avg_train_loss)
    print('Training loss:', avg_train_loss, file=f_1, flush=True)
    
    # validation phase
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            
            # mask 15% of the label tokens for prediction
            input_ids, labels = pth.mask_tokens(input_ids, tokenizer, mlm_probability=0.15)
            
            #input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean()
            
            val_loss += loss.item()
            num_batches += 1
            
    avg_val_loss = val_loss / num_batches
    print(f"Epoch {epoch + 1}/{epochs} validation completed. Validation Loss: {avg_val_loss}", flush=True)
    avg_val_loss_ls.append(avg_val_loss)
    print('Validation loss:', avg_val_loss, file=f_1)
    
    # save the model only if valid loss decreases
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # save the trained model
        model.module.save_pretrained("saved_models")
        tokenizer.save_pretrained("saved_models")
        print(f"Model from epoch {epoch + 1} is saved", flush=True)
        
        
# write loss values to file
#f_1 = open('loss_vals.txt', 'w')
print('Training loss list:', avg_train_loss_ls, file=f_1)
print('Validation loss list:', avg_val_loss_ls, file=f_1)
f_1.close()

# make graph of loss values
epoch = list(np.arange(epochs) + 1)
plt.plot(epoch, avg_train_loss_ls, label='Training Loss')
plt.plot(epoch, avg_val_loss_ls, label='Validation Loss')

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('Training and Validation Loss Plot (lr='+str(lr)+' with linear decay)')
plt.ylim(5, 6.5)
plt.savefig('loss_'+str(lr)+'_linear_decay.jpg',format='jpeg', bbox_inches='tight', dpi=300)

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
