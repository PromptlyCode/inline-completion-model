import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Constants
MAX_LENGTH = 20
HIDDEN_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# Helper function to convert numbers to words
def number_to_words(n):
    ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    
    if n == 0:
        return 'zero'
    
    def recurse(n):
        if n == 0:
            return []
        elif n < 10:
            return [ones[n]]
        elif n < 20:
            return [teens[n-10]]
        elif n < 100:
            return [tens[n//10]] + ([ones[n%10]] if n%10 != 0 else [])
        else:
            return [ones[n//100], 'hundred'] + (recurse(n%100) if n%100 != 0 else [])
    
    return ' '.join(recurse(n))

# Generate training data
def generate_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        num = random.randint(0, 999)
        data.append((str(num), number_to_words(num)))
    return data

# Custom dataset class
class NumberDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.char_to_idx = {char: idx for idx, char in enumerate('0123456789 ')}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        word_set = set()
        for _, target in data:
            word_set.update(target.split())
        for idx, word in enumerate(sorted(word_set)):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source, target = self.data[idx]
        source_tensor = torch.zeros(MAX_LENGTH)
        for i, char in enumerate(source):
            source_tensor[i] = self.char_to_idx[char]
            
        target_words = target.split()
        target_tensor = torch.tensor([self.word_to_idx[word] for word in target_words])
        return source_tensor, target_tensor

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, x):
        embedded = self.embedding(x.long())
        output, hidden = self.gru(embedded.view(1, 1, -1))
        return output, hidden

# Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        output = self.embedding(x.long()).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

# Main Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        target_len = target.size(0)
        batch_size = 1
        target_vocab_size = self.decoder.out.out_features
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)
        
        encoder_output, hidden = self.encoder(source)
        
        decoder_input = torch.tensor([0])  # Start token
        
        for t in range(target_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = target[t] if teacher_force else top1
            
        return outputs

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for source, target in train_loader:
        source, target = source.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(source[0], target[0])
        
        loss = criterion(output.view(-1, output.size(-1)), target.long())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

# Visualization function
def plot_training_progress(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def main():
    # Generate data
    data = generate_data()
    dataset = NumberDataset(data)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(len(dataset.char_to_idx), HIDDEN_SIZE)
    decoder = Decoder(HIDDEN_SIZE, len(dataset.word_to_idx))
    model = Seq2Seq(encoder, decoder).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    losses = []
    for epoch in range(NUM_EPOCHS):
        loss = train(model, train_loader, criterion, optimizer, device)
        losses.append(loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss:.4f}')
    
    # Save model
    save_model(model, 'number_translator.pth')
    
    # Visualize training progress
    plot_training_progress(losses)

if __name__ == '__main__':
    main()

