import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import tokenize
from io import StringIO
import os

# Tokenizer for Python code
class CodeTokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.token2idx = {}
        self.idx2token = {}
        self.vocab_freq = {}
        
    def fit(self, code_files):
        # Collect all tokens from code files
        for file in code_files:
            with open(file, 'r', encoding='utf-8') as f:
                code = f.read()
                tokens = self._tokenize_code(code)
                for token in tokens:
                    self.vocab_freq[token] = self.vocab_freq.get(token, 0) + 1
        
        # Build vocabulary
        sorted_tokens = sorted(self.vocab_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (token, _) in enumerate(sorted_tokens[:self.vocab_size-2]):
            self.token2idx[token] = idx + 2
            self.idx2token[idx + 2] = token
        
        # Add special tokens
        self.token2idx['<PAD>'] = 0
        self.token2idx['<UNK>'] = 1
        self.idx2token[0] = '<PAD>'
        self.idx2token[1] = '<UNK>'
    
    def _tokenize_code(self, code):
        tokens = []
        try:
            for tok in tokenize.generate_tokens(StringIO(code).readline):
                if tok.string.strip():
                    tokens.append(tok.string)
        except:
            pass
        return tokens
    
    def encode(self, code):
        tokens = self._tokenize_code(code)
        return [self.token2idx.get(token, 1) for token in tokens]  # 1 is <UNK>
    
    def decode(self, indices):
        return ' '.join(self.idx2token.get(idx, '<UNK>') for idx in indices)

# Dataset
class CodeDataset(Dataset):
    def __init__(self, code_files, tokenizer, seq_length=512):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []
        
        for file in code_files:
            with open(file, 'r', encoding='utf-8') as f:
                code = f.read()
                tokens = self.tokenizer.encode(code)
                
                # Create sequences
                for i in range(0, len(tokens) - seq_length):
                    input_seq = tokens[i:i + seq_length]
                    target_seq = tokens[i + 1:i + seq_length + 1]
                    self.data.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)

# Transformer Model
class CodeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Training function
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}')

# Code completion function
def complete_code(model, tokenizer, input_code, max_length=50, temperature=0.8):
    model.eval()
    tokens = tokenizer.encode(input_code)
    input_tensor = torch.tensor(tokens).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor)
            next_token_logits = output[0, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.token2idx.get('<PAD>'):
                break
    
    return tokenizer.decode(input_tensor[0].tolist())

# Main execution
def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    code_files = [str(p) for p in Path('./your_code_directory').glob('**/*.py')]
    
    # Initialize tokenizer and create dataset
    tokenizer = CodeTokenizer()
    tokenizer.fit(code_files)
    
    dataset = CodeDataset(code_files, tokenizer)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = CodeTransformer(len(tokenizer.token2idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train model
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Save model
    torch.save(model.state_dict(), 'code_completion_model.pth')
    
    # Example usage
    input_code = "def hello_world():"
    completed_code = complete_code(model, tokenizer, input_code)
    print(f"Input: {input_code}")
    print(f"Completed: {completed_code}")

if __name__ == "__main__":
    main()

