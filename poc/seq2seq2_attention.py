import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import pickle

class Dictionary:
    def __init__(self):
        self.word2idx = {'<pxad>': 0, '<bxos>': 1, '<exos>': 2}
        self.idx2word = {0: '<pxad>', 1: '<bxos>', 2: '<exos>'}
        self.word_count = Counter()
        self.vocab_size = 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
        self.word_count[word] += 1

    def __len__(self):
        return self.vocab_size

class TranslationDataset(Dataset):
    def __init__(self, en_path, zh_path, max_length=50):
        self.en_dict = Dictionary()
        self.zh_dict = Dictionary()
        self.max_length = max_length
        
        # Load and process data
        with open(en_path, 'r', encoding='utf-8') as f:
            self.en_sentences = f.readlines()
        with open(zh_path, 'r', encoding='utf-8') as f:
            self.zh_sentences = f.readlines()
            
        # Build vocabularies
        for sent in self.en_sentences:
            for word in sent.strip().split():
                self.en_dict.add_word(word)
        
        for sent in self.zh_sentences:
            for char in sent.strip():
                self.zh_dict.add_word(char)

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en_sent = ['<bxos>'] + self.en_sentences[idx].strip().split() + ['<exos>']
        zh_sent = ['<bxos>'] + list(self.zh_sentences[idx].strip()) + ['<exos>']
        
        # Convert to indices and pad
        en_indices = [self.en_dict.word2idx[w] for w in en_sent]
        zh_indices = [self.zh_dict.word2idx[w] for w in zh_sent]
        
        # Pad sequences
        en_indices = en_indices[:self.max_length]
        zh_indices = zh_indices[:self.max_length]
        
        en_indices += [self.en_dict.word2idx['<pxad>']] * (self.max_length - len(en_indices))
        zh_indices += [self.zh_dict.word2idx['<pxad>']] * (self.max_length - len(zh_indices))
        
        return torch.LongTensor(en_indices), torch.LongTensor(zh_indices)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, 
                           dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src):
        # src shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, embed_size]
        
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs shape: [batch_size, seq_len, hidden_size * 2]
        # hidden shape: [n_layers * 2, batch_size, hidden_size]
        
        # Convert bidirectional hidden states
        hidden = self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        cell = self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1))
        
        hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.n_layers, 1, 1)
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        
    def forward(self, hidden, encoder_outputs):
        # hidden shape: [batch_size, hidden_size]
        # encoder_outputs shape: [batch_size, seq_len, hidden_size * 2]
        
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        # Repeat hidden state seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate hidden state with encoder outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Apply attention vector
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(energy, v.transpose(1, 2)).squeeze(2)
        
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_size, embed_size)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_size + hidden_size * 2, hidden_size, n_layers,
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size * 3, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input shape: [batch_size]
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_size]
        
        # Calculate attention weights
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # Apply attention to encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_size * 2]
        
        # Combine embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        # Combine output and context for final prediction
        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        prediction = self.fc(output)
        
        return prediction, hidden, cell

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs

def save_model(model, en_dict, zh_dict, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'en_dict': en_dict,
        'zh_dict': zh_dict
    }
    torch.save(checkpoint, path)

def load_model(path, device):
    checkpoint = torch.load(path)
    en_dict = checkpoint['en_dict']
    zh_dict = checkpoint['zh_dict']
    
    # Create model with the same architecture as training
    encoder = Encoder(len(en_dict), 256, 512, n_layers=2, dropout=0.1)
    decoder = Decoder(len(zh_dict), 256, 512, n_layers=2, dropout=0.1)
    model = Seq2SeqWithAttention(encoder, decoder, device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model, en_dict, zh_dict

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (src, trg) in enumerate(train_loader):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)


def main():
    # Hyperparameters
    BATCH_SIZE = 32
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    N_LAYERS = 2
    DROPOUT = 0.1
    N_EPOCHS = 50
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = TranslationDataset('../synthetic_data/news-commentary-v12.zh-en.en',
                               '../synthetic_data/news-commentary-v12.zh-en.zh')
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    encoder = Encoder(len(dataset.en_dict), EMBED_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(device)
    decoder = Decoder(len(dataset.zh_dict), EMBED_SIZE, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(device)
    model = Seq2SeqWithAttention(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.zh_dict.word2idx['<pxad>'])

    # Training loop
    for epoch in range(N_EPOCHS):
        loss = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')
        save_model(model, dataset.en_dict, dataset.zh_dict, 'translation_model.pt')

    # Save model
    save_model(model, dataset.en_dict, dataset.zh_dict, 'translation_model.pt')

def predict(model, en_dict, zh_dict, sentence, device, max_length=50):
    model.eval()

    # Tokenize and convert to indices
    tokens = ['<bxos>'] + sentence.strip().split() + ['<exos>']
    src_indices = [en_dict.word2idx.get(token, en_dict.word2idx['<bxos>']) for token in tokens]

    # Pad sequence
    src_indices = src_indices[:max_length]
    src_indices += [en_dict.word2idx['<pxad>']] * (max_length - len(src_indices))

    # Convert to tensor
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    # Get encoder outputs
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    # Initialize decoder input
    decoder_input = torch.LongTensor([zh_dict.word2idx['<bxos>']]).to(device)

    # Store predictions
    predictions = ['<bxos>']

    # Decode one token at a time
    for _ in range(max_length):
        with torch.no_grad():
            output, hidden, cell = model.decoder(decoder_input, hidden, cell, encoder_outputs)

        # Get predicted token
        pred_token_idx = output.argmax(1).item()
        pred_token = zh_dict.idx2word[pred_token_idx]

        # Break if end token
        if pred_token == '<exos>':
            break

        predictions.append(pred_token)
        decoder_input = torch.LongTensor([pred_token_idx]).to(device)

    # Remove special tokens and join characters
    translation = ''.join([char for char in predictions[1:] if char not in ['<bxos>', '<exos>', '<pxad>']])

    return translation

# Example usage:
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, en_dict, zh_dict = load_model('translation_model.pt', device)
english_sentence = "Hello, how are you?"
chinese_translation = predict(model, en_dict, zh_dict, english_sentence, device)
print(f"English: {english_sentence}")
print(f"Chinese: {chinese_translation}")
"""

if __name__ == '__main__':
    main()

