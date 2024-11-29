import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter
import jieba

# Define constants
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
MAX_LENGTH = 100

# Add these constants at the top with other hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
BATCH_SIZE = 64  # Increased from default
TEACHER_FORCING_RATIO = 0.5

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1
                
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

class TranslationDataset(Dataset):
    def __init__(self, english_data, chinese_data, eng_vocab, chi_vocab):
        self.english_data = english_data
        self.chinese_data = chinese_data
        self.eng_vocab = eng_vocab
        self.chi_vocab = chi_vocab
        
    def __len__(self):
        return len(self.english_data)
    
    def __getitem__(self, index):
        eng_text = self.english_data[index]
        chi_text = self.chinese_data[index]
        
        # Convert to indices
        eng_numericalized = [self.eng_vocab.stoi["<bos>"]]
        eng_numericalized += [self.eng_vocab.stoi.get(token, self.eng_vocab.stoi["<unk>"]) 
                            for token in eng_text]
        eng_numericalized.append(self.eng_vocab.stoi["<eos>"])
        
        chi_numericalized = [self.chi_vocab.stoi["<bos>"]]
        chi_numericalized += [self.chi_vocab.stoi.get(token, self.chi_vocab.stoi["<unk>"]) 
                            for token in chi_text]
        chi_numericalized.append(self.chi_vocab.stoi["<eos>"])
        
        return torch.tensor(eng_numericalized), torch.tensor(chi_numericalized)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (seq_length, batch_size)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden, cell):
        # x shape: (batch_size)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

# Update the Seq2Seq forward method to include teacher forcing
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        # Fix: Use the size of the decoder's final linear layer instead of output_dim
        target_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        hidden, cell = self.encoder(source)

        # First input to decoder is the <bos> token
        input = target[0,:]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = target[t] if teacher_force else top1

        return outputs

def save_model(model, optimizer, eng_vocab, chi_vocab, epoch, loss, filename):
    """
    Save the model checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eng_vocab': eng_vocab,
        'chi_vocab': chi_vocab,
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

def load_model(filename, device):
    """
    Load the model checkpoint
    """
    checkpoint = torch.load(filename, map_location=device)

    # Initialize model with same architecture
    encoder = Encoder(len(checkpoint['eng_vocab']), EMBEDDING_SIZE, HIDDEN_SIZE,
                     NUM_LAYERS, DROPOUT)
    decoder = Decoder(len(checkpoint['chi_vocab']), EMBEDDING_SIZE, HIDDEN_SIZE,
                     len(checkpoint['chi_vocab']), NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Load model parameters
    model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['eng_vocab'], checkpoint['chi_vocab'], \
           checkpoint['epoch'], checkpoint['loss']

def translate_sentence(model, sentence, eng_vocab, chi_vocab, device, max_length=50):
    """
    Translate a single English sentence to Chinese
    """
    model.eval()

    if isinstance(sentence, str):
        tokens = sentence.split()
    else:
        tokens = sentence

    # Convert tokens to indices
    tokens = [eng_vocab.stoi.get(token, eng_vocab.stoi['<unk>']) for token in tokens]
    tokens = [eng_vocab.stoi['<bos>']] + tokens + [eng_vocab.stoi['<eos>']]

    source = torch.LongTensor(tokens).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(source)

    outputs = [chi_vocab.stoi['<bos>']]

    for _ in range(max_length):
        previous = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        if best_guess == chi_vocab.stoi['<eos>']:
            break

    translated_tokens = [chi_vocab.itos[idx] for idx in outputs]
    # Remove special tokens
    translated_tokens = translated_tokens[1:-1]  # Remove <bos> and <eos>
    return ''.join(translated_tokens)  # Join without spaces for Chinese

def collate_fn(batch):
    """
    Custom collate function for DataLoader
    """
    # Separate source and target sequences
    src_batch, tgt_batch = [], []
    for src_item, tgt_item in batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)

    # Pad sequences
    src_batch = pad_sequence(src_batch)
    tgt_batch = pad_sequence(tgt_batch)

    return src_batch, tgt_batch

def train_model():
    # Load and preprocess data
    english_data = open("../synthetic_data/news-commentary-v12.zh-en.en").readlines()[1:100000]
    chinese_data = open("../synthetic_data/news-commentary-v12.zh-en.zh").readlines()[1:100000]

    # Tokenize data
    english_tokenized = [sentence.strip().split() for sentence in english_data]
    chinese_tokenized = [list(jieba.cut(sentence.strip())) for sentence in chinese_data]

    # Create vocabularies
    eng_vocab = Vocabulary()
    chi_vocab = Vocabulary()
    eng_vocab.build_vocabulary(english_tokenized)
    chi_vocab.build_vocabulary(chinese_tokenized)

    # Create dataset and dataloader
    dataset = TranslationDataset(english_tokenized, chinese_tokenized, eng_vocab, chi_vocab)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,  # Use custom collate function
        drop_last=True
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(len(eng_vocab), EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    decoder = Decoder(len(chi_vocab), EMBEDDING_SIZE, HIDDEN_SIZE, len(chi_vocab), NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Define optimizer with specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Define loss function with label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=eng_vocab.stoi["<pad>"],
        label_smoothing=0.1  # Add label smoothing to prevent overconfidence
    )

    # Training loop
    best_loss = float('inf')
    early_stopping_patience = 5
    early_stopping_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        batch_losses = []

        for batch_idx, (source, target) in enumerate(loader):
            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(source, target, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())

            if batch_idx % 50 == 0:  # Print more frequently
                avg_batch_loss = sum(batch_losses[-50:]) / len(batch_losses[-50:]) if batch_losses else 0
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
                      f"Batch [{batch_idx}/{len(loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Avg Loss: {avg_batch_loss:.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(avg_loss)

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stopping_counter = 0
            save_model(model, optimizer, eng_vocab, chi_vocab, epoch, avg_loss,
                      'best_translator.pth')
            print(f"New best model saved with loss: {avg_loss:.4f}")
        else:
            early_stopping_counter += 1

        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, eng_vocab, chi_vocab, epoch, avg_loss,
                      f'translator_checkpoint_epoch_{epoch+1}.pth')

def main():
    # Training
    train_model()

    # Loading and testing the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, eng_vocab, chi_vocab, epoch, loss = load_model('best_translator.pth', device)

    # Test translation
    test_sentence = "The weather is very nice today."
    translated = translate_sentence(model, test_sentence, eng_vocab, chi_vocab, device)
    print(f"English: {test_sentence}")
    print(f"Chinese: {translated}")

if __name__ == "__main__":
    main()

