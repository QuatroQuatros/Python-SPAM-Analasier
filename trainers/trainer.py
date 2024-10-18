from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from dataset.legit_emails import legitimate_emails

df_frauds = pd.read_csv('dataset/frauds_dataset.csv')

fraudulent_subjects = df_frauds['subject'].tolist()
fraudulent_contents = df_frauds['content'].tolist()
fraudulent_emails = [f"{sub} [SEP] {cont}" for sub, cont in zip(fraudulent_subjects, fraudulent_contents)]

fraudulent_labels = [1] * len(fraudulent_emails)

legitimate_subjects = [email["subject"] for email in legitimate_emails]
legitimate_contents = [email["content"] for email in legitimate_emails]
legitimate_emails_combined = [f"{sub} [SEP] {cont}" for sub, cont in zip(legitimate_subjects, legitimate_contents)]

legitimate_labels = [0] * len(legitimate_emails_combined)

# Combinar e-mails e labels
all_emails = legitimate_emails_combined + fraudulent_emails
all_labels = legitimate_labels + fraudulent_labels

train_texts, val_texts, train_labels, val_labels = train_test_split(all_emails, all_labels, test_size=0.2, random_state=42)

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs, torch.tensor(label)

tokenizer = AutoTokenizer.from_pretrained("Titeiiko/OTIS-Official-Spam-Model")

# Criar os datasets de treino e validação
train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
val_dataset = SpamDataset(val_texts, val_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Carregar o modelo
model = AutoModelForSequenceClassification.from_pretrained("Titeiiko/OTIS-Official-Spam-Model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=4e-5)
loss_fn = CrossEntropyLoss()

model.train()

num_epochs = 3

for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {loss.item()}")

    # Validar após cada época
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss after epoch {epoch + 1}: {avg_val_loss}")
    model.train()

model.save_pretrained("../models")
tokenizer.save_pretrained("../tokenizer")
