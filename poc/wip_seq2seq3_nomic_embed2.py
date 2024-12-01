import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests

def embed_code(code_snippets):
    """Embed code using Ollama Nomic Embed Text."""
    embeddings = []
    for snippet in code_snippets:
        response = requests.post('http://localhost:11434/api/embeddings',
                               json={
                                   "model": "nomic-embed-text",
                                   "prompt": snippet
                               })
        if response.status_code == 200:
            embedding = response.json()['embedding']
            embeddings.append(embedding)
        else:
            print(f"Error embedding snippet: {response.status_code}")
            embeddings.append([0] * 768)
    return np.array(embeddings, dtype=np.float32)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Remove embedding layer since we're using pre-computed embeddings
        self.input_layer = nn.Linear(768, hidden_dim)  # 768 is the Nomic embedding dimension
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src len, batch size, 768]
        transformed = self.dropout(self.input_layer(src))
        # transformed = [src len, batch size, hidden dim]
        
        outputs, (hidden, cell) = self.rnn(transformed)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            
        return outputs

class TranslationDataset(Dataset):
    def __init__(self, en_sentences, cn_sentences, cn_vocab):
        self.en_sentences = en_sentences
        self.cn_sentences = cn_sentences
        self.cn_vocab = cn_vocab
        
        # Pre-compute embeddings for all English sentences
        self.en_embeddings = embed_code(en_sentences)
        
    def __len__(self):
        return len(self.en_sentences)
    
    def __getitem__(self, idx):
        en_emb = torch.tensor(self.en_embeddings[idx], dtype=torch.float32)
        
        cn_sent = ['<bxos>'] + self.cn_sentences[idx].split() + ['<exos>']
        cn_indices = [self.cn_vocab[token] if token in self.cn_vocab else self.cn_vocab['<pxad>'] 
                     for token in cn_sent]
        
        return en_emb, torch.tensor(cn_indices)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        # Modify src shape to [seq_len, batch_size, embedding_dim]
        src = src.unsqueeze(0) if src.dim() == 2 else src
        trg = trg.transpose(0, 1)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.unsqueeze(0) if src.dim() == 2 else src
            trg = trg.transpose(0, 1)
            
            output = model(src, trg, 0)
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Example usage:
"""
# Initialize Chinese vocabulary
cn_vocab = {'<pxad>': 0, '<bxos>': 1, '<exos>': 2, ...}  # Add Chinese words

# Model parameters
OUTPUT_DIM = len(cn_vocab)
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc = Encoder(None, HID_DIM, N_LAYERS, ENC_DROPOUT)  # input_dim not needed
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# Training parameters
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=cn_vocab['<pxad>'])
CLIP = 1

# Create dataset and dataloader
dataset = TranslationDataset(en_sentences, cn_sentences, cn_vocab)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss = train(model, dataloader, optimizer, criterion, CLIP)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}')
"""

