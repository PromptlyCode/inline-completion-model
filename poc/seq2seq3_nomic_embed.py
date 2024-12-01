import requests
import numpy as np

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

import torch
import torch.nn as nn
import numpy as np
import requests
from torch.utils.data import Dataset, DataLoader

# Special tokens
PAD_token = '<pxad>'
BOS_token = '<bxos>'
EOS_token = '<exos>'

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)  # Using linear layer for custom embeddings
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src_len, batch_size, input_dim]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        # input = [batch_size]
        # hidden = [n_layers, batch_size, hidden_dim]
        input = input.unsqueeze(0)
        # input = [1, batch_size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        output, hidden = self.rnn(embedded, hidden)
        # output = [1, batch_size, hidden_dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size, input_dim]
        # trg = [trg_len, batch_size]
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        
        input = trg[0,:]  # First input is <BOS> token
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            
        return outputs

class TranslationDataset(Dataset):
    def __init__(self, en_texts, zh_texts, vocab_zh):
        self.embeddings = embed_code(en_texts)
        self.zh_texts = zh_texts
        self.vocab_zh = vocab_zh
        
    def __len__(self):
        return len(self.zh_texts)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.embeddings[idx]), 
                torch.LongTensor([self.vocab_zh[token] for token in self.zh_texts[idx].split()]))

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(train_loader):
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)

# Example usage
def main():
    # Hyperparameters
    INPUT_DIM = 768  # Dimension of Nomic embeddings
    OUTPUT_DIM = len(vocab_zh)  # Size of Chinese vocabulary
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_zh[PAD_token])
    
    # Create dataset and dataloader
    dataset = TranslationDataset(en_texts, zh_texts, vocab_zh)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    N_EPOCHS = 10
    for epoch in range(N_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}')

if __name__ == "__main__":
    main()

