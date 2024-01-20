
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 

# Step 1: Load Data
traindf = pd.concat([pd.read_csv('/home/lucas/proj/dhlab/data/cam_train.csv'),pd.read_csv('/home/lucas/proj/dhlab/data/hand_train.csv')])
validdf = pd.read_csv('data/validation_lucas.csv')
traindf.dropna(inplace=True, subset=['summary'])
validdf['label'] = validdf['labels']
trainds = traindf[['summary', 'label']]
valids = validdf[['summary', 'label']]
#reset index
valids = valids.reset_index(drop=True)
trainds = trainds.reset_index(drop=True)




# Assuming traindf and validdf are your dataframes
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
            pad_to_max_length=True,
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

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Dataset
training_set = TextDataset(trainds, tokenizer)
validation_set = TextDataset(valids, tokenizer)

# Step 3: DataLoader
train_params = {'batch_size': 32, 'shuffle': True}
valid_params = {'batch_size': 32, 'shuffle': False}

train_loader = DataLoader(training_set, **train_params)
valid_loader = DataLoader(validation_set, **valid_params)

# Step 4: Load BERT Model
from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=1)
model.to('cuda')

# Step 5: Training
optimizer = AdamW(model.parameters(), lr=1e-5)

def train(epoch):
    model.train()
    total_batches = len(train_loader)
    for batch_num, data in enumerate(train_loader, 0):
        ids = data['ids'].to('cuda')
        mask = data['mask'].to('cuda')
        targets = data['targets'].to('cuda').unsqueeze(1)  # Reshape targets

        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, targets)

        loss.backward()
        optimizer.step()

        # Print progress
        print(f'Epoch: {epoch}, Batch: {batch_num + 1}/{total_batches}, Loss: {loss.item()}')

    
def validate(epoch):
    model.eval()
    total_loss, total_accuracy = 0, 0

    for _, data in enumerate(valid_loader, 0):
        ids = data['ids'].to('cuda')
        mask = data['mask'].to('cuda')
        targets = data['targets'].to('cuda')

        with torch.no_grad():
            outputs = model(ids, mask, labels=targets)
            loss = outputs.loss
            logits = outputs.logits

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = (preds == targets).cpu().numpy().mean() * 100
        total_accuracy += accuracy

    avg_loss = total_loss / len(valid_loader)
    avg_accuracy = total_accuracy / len(valid_loader)
    print(f'Validation Results - Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {avg_accuracy}%')


for epoch in range(3):  # Number of epochs can be adjusted
    train(epoch)
    validate(epoch)


# Step 7: Save the Model
model.save_pretrained('/home/lucas/proj/dhlab/data/model/hand_model')
