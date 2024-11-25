import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from typing import List, Tuple
import numpy as np
from collections import defaultdict
import os
import re

# Token processing and vocabulary
class Vocab:
    def __init__(self, pad_token="<pad>", unk_token="<unk>", sos_token="<sos>", eos_token="<eos>"):
        self.token2idx = {}
        self.idx2token = {}
        self.token_freq = defaultdict(int)
        
        # Special tokens
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        # Add special tokens to vocabulary
        for token in [pad_token, unk_token, sos_token, eos_token]:
            self.add_token(token)
    
    def add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        self.token_freq[token] += 1
    
    def __len__(self):
        return len(self.token2idx)

# Code tokenizer
class CodeTokenizer:
    def __init__(self):
        self.pattern = re.compile(r'([a-zA-Z_]\w*|[0-9]+|\S)')
    
    def tokenize(self, code: str) -> List[str]:
        return self.pattern.findall(code)

# Seq2Seq Model Architecture
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size * 2, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 3, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(input))  # (batch_size, 1, embed_size)
        
        # Attention
        attention = torch.bmm(encoder_outputs, hidden[-1].unsqueeze(2))
        attention_weights = torch.softmax(attention, dim=1)
        context = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)
        
        # Combine embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        # Combine output and context for prediction
        prediction = torch.cat((output, context), dim=2)
        prediction = self.fc_out(prediction.squeeze(1))
        
        return prediction, hidden, cell, attention_weights

class Seq2SeqCodeCompletion(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio: float = 0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # First input to decoder is <sos> token
        decoder_input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = trg[:, t] if teacher_force else top1
            
        return outputs

# Data preparation
class CodeDataset:
    def __init__(self, code_files: List[str], vocab: Vocab, tokenizer: CodeTokenizer, 
                 max_length: int = 512):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._process_files(code_files)
        
    def _process_files(self, code_files: List[str]) -> List[Tuple[List[int], List[int]]]:
        examples = []
        for file_path in code_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            tokens = self.tokenizer.tokenize(code)
            
            # Create sliding windows of tokens
            for i in range(0, len(tokens) - self.max_length, self.max_length // 2):
                window = tokens[i:i + self.max_length]
                input_tokens = [self.vocab.sos_token] + window[:-1]
                target_tokens = window + [self.vocab.eos_token]
                
                input_ids = [self.vocab.token2idx.get(t, self.vocab.token2idx[self.vocab.unk_token]) 
                            for t in input_tokens]
                target_ids = [self.vocab.token2idx.get(t, self.vocab.token2idx[self.vocab.unk_token]) 
                             for t in target_tokens]
                
                examples.append((input_ids, target_ids))
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# Training function
def train(model: Seq2SeqCodeCompletion, iterator, optimizer, criterion, clip: float = 1.0):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        src, src_lengths, trg = batch
        
        optimizer.zero_grad()
        output = model(src, src_lengths, trg)
        
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Inference function
def generate_completion(model: Seq2SeqCodeCompletion, src_tokens: List[str], vocab: Vocab, 
                       max_length: int = 50):
    model.eval()
    
    # Convert tokens to indices
    src_indices = [vocab.token2idx.get(t, vocab.token2idx[vocab.unk_token]) for t in src_tokens]
    src_tensor = torch.LongTensor([src_indices]).to(model.device)
    src_lengths = torch.LongTensor([len(src_indices)])
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lengths)
        decoder_input = torch.LongTensor([vocab.token2idx[vocab.sos_token]]).to(model.device)
        
        completed_tokens = []
        for _ in range(max_length):
            output, hidden, cell, _ = model.decoder(decoder_input, hidden, cell, encoder_outputs)
            top1 = output.argmax(1)
            
            if top1.item() == vocab.token2idx[vocab.eos_token]:
                break
                
            completed_tokens.append(vocab.idx2token[top1.item()])
            decoder_input = top1
    
    return completed_tokens

# Example usage
def main():
    # Hyperparameters
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.1
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    
    # Initialize components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CodeTokenizer()
    vocab = Vocab()
    
    # Process your code files and build vocabulary
    code_files = ['/home/xlisp/EmacsPyPro/jim-emacs-fun-py']
    dataset = CodeDataset(code_files, vocab, tokenizer)
    
    # Create model
    encoder = Encoder(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    decoder = Decoder(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    model = Seq2SeqCodeCompletion(encoder, decoder, device).to(device)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx[vocab.pad_token])
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, dataset, optimizer, criterion)
        print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}')
    
    # Example completion
    code_snippet = "def calculate_sum(a, b):"
    tokens = tokenizer.tokenize(code_snippet)
    completed_tokens = generate_completion(model, tokens, vocab)
    completed_code = ' '.join(completed_tokens)
    print(f'Input: {code_snippet}')
    print(f'Completion: {completed_code}')

if __name__ == "__main__":
    main()

