#!/usr/bin/env python3

import argparse
import pandas as pd
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torch.cuda.amp import GradScaler, autocast
import os

YEARS_TO_REMOVE = 5 

"""
This script is used to remove years from the training set and evaluate the model on the validation set.

"""




class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.data.summary[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.label[index], dtype=torch.float)
        }

    def __len__(self):
        return self.len

def train(epoch , model,optimizer, train_loader,is_fp16=False):
    model.train()
    total_batches = len(train_loader)
    loop =tqdm(enumerate(train_loader, 0), total=len(train_loader), desc='Training')
    if is_fp16: # Use mixed precision
        scaler = GradScaler()
        for batch_num, data in loop :
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            targets = data['targets'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(ids, mask)
                loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_description(f'Epoch: {epoch}, Batch: {batch_num + 1}/{total_batches}, Loss: {loss.item()}')
            # Print progress
            
        
    else:
        for batch_num, data in loop :
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            targets = data['targets'].to(device).unsqueeze(1)  # Reshape targets

            optimizer.zero_grad()
            outputs = model(ids, mask)
            loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, targets)

            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch: {epoch}, Batch: {batch_num + 1}/{total_batches}, Loss: {loss.item()}')
            # Print progress

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    total_batches = len(val_loader)
    loop = tqdm(enumerate(val_loader), total=total_batches, leave=False)

    with torch.no_grad():
        for batch_num, (images, targets) in loop:
            images, targets = images.to(device), targets.to(device)
            total_samples += targets.size(0)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Apply sigmoid if the model outputs logits
            probabilities = torch.sigmoid(outputs)

            # Convert probabilities to predicted class (0 or 1)
            predicted = probabilities >= 0.5

            # Make sure that the shapes of predictions and targets are consistent
            predicted = predicted.view_as(targets)

            correct_predictions += (predicted == targets).sum().item()

            # Update tqdm loop
            loop.set_description(f'Epoch: {epoch}, Loss: {loss.item()}')

    avg_loss = total_loss / total_batches
    accuracy = correct_predictions / total_samples
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')


def run_inference(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader),desc='Inference'):
            input_ids = batch['ids'].to(device)
            attention_mask = batch['mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs[:, 0])  
    
    return predictions


def main():
# Write to Excel
    file_path = f'/mnt/d/proj/dhlab/models/{model_name}/classification_reports_{seed}.xlsx'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    removed_years = sorted(list(set(traindf['year'])),reverse=True)[:YEARS_TO_REMOVE+1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with pd.ExcelWriter(file_path) as writer:
        for i in range(YEARS_TO_REMOVE+1):
            year = removed_years[i]
            print(f'Year: {year}')
            trainds = traindf[traindf.year <=year]
            validds = validdf
            trainds = trainds.reset_index(drop=True)
            validds = validds.reset_index(drop=True)
            # Step 4:  model
            
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)


            # Step 5: Create dataloaders
            training_set = TextDataset(trainds, tokenizer)
            validation_set = TextDataset(validds, tokenizer)

            train_params = {'batch_size':  batch_size,
                            'shuffle': True,
                            }

            valid_params = {'batch_size': 32,
                            'shuffle': False,
                            }

            training_loader = DataLoader(training_set, **train_params)
            validation_loader = DataLoader(validation_set, **valid_params)

            # Step 6: Train and validate
            optimizer = AdamW(model.parameters(), lr=1e-5)
            epochs = 3 if not (fp16 ) else 5
            for epoch in range(1, epochs + 1):
                train(epoch, model, optimizer, training_loader, is_fp16=fp16)
                validate(epoch, model, validation_loader, torch.nn.BCEWithLogitsLoss(), device)
            
            # Step 7: evaluate on validation set and save to csv
            validds['pred'] = [1 if x> 0.5 else 0 for x in run_inference(model, validation_loader)]
            report_df = pd.DataFrame(classification_report(validds['label'], validds['pred'], output_dict=True,digits=4)).T.round(4)
            report_df.to_excel(writer, sheet_name=f'cutttoff_{year}')
            
            # Save model
            model.save_pretrained( f'/mnt/d/proj/dhlab/models/{model_name}/seed{seed}/{i}_years_removed')
    

        
            del model  # Delete the model
            torch.cuda.empty_cache()  # Clear cache

            print(f'Model saved for year {year}')

if __name__ == '__main__':
    # Step 1: Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, help='Seed value')
    parser.add_argument('model_name', type=str, help='Model name')
    parser.add_argument('--fp16', help='Whether to use fp16', action='store_true')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    args = parser.parse_args()

    # Access the command line arguments
    seed =args.seed
    torch.cuda.manual_seed(seed)
    model_name = args.model_name
    fp16 = args.fp16
    batch_size = args.batch_size
    # Step 2: Load Data
    traindf = pd.read_pickle('/home/lucas/proj/dhlab/data/datasets/final_train_with_years.pkl')
    validdf = pd.read_pickle('/home/lucas/proj/dhlab/data/datasets/valid_with_years.pkl')
    main()