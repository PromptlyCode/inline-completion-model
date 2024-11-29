import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import Counter
import jieba

# Define constants
BATCH_SIZE = 32
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
MAX_LENGTH = 100

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<pxad>", 1: "<bxos>", 2: "<exos>", 3: "<unk>"}
        self.stoi = {"<pxad>": 0, "<bxos>": 1, "<exos>": 2, "<unk>": 3}
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
        eng_numericalized = [self.eng_vocab.stoi["<bxos>"]]
        eng_numericalized += [self.eng_vocab.stoi.get(token, self.eng_vocab.stoi["<unk>"]) 
                            for token in eng_text]
        eng_numericalized.append(self.eng_vocab.stoi["<exos>"])
        
        chi_numericalized = [self.chi_vocab.stoi["<bxos>"]]
        chi_numericalized += [self.chi_vocab.stoi.get(token, self.chi_vocab.stoi["<unk>"]) 
                            for token in chi_text]
        chi_numericalized.append(self.chi_vocab.stoi["<exos>"])
        
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

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        hidden, cell = self.encoder(source)
        
        x = target[0]  # Start token
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if torch.rand(1).item() < teacher_force_ratio else best_guess
            
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
    tokens = [eng_vocab.stoi['<bxos>']] + tokens + [eng_vocab.stoi['<exos>']]

    source = torch.LongTensor(tokens).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(source)

    outputs = [chi_vocab.stoi['<bxos>']]

    for _ in range(max_length):
        previous = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        if best_guess == chi_vocab.stoi['<exos>']:
            break

    translated_tokens = [chi_vocab.itos[idx] for idx in outputs]
    # Remove special tokens
    translated_tokens = translated_tokens[1:-1]  # Remove <bxos> and <exos>
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
    english_data = open("../synthetic_data/news-commentary-v12.zh-en.en").readlines()
    chinese_data = open("../synthetic_data/news-commentary-v12.zh-en.zh").readlines()

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

    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=eng_vocab.stoi["<pxad>"])

    # Training loop
    num_epochs = 10
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (source, target) in enumerate(loader):
            source = source.to(device)
            target = target.to(device)

            output = model(source, target)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss/len(loader)
        print(f"Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, eng_vocab, chi_vocab, epoch, avg_loss,
                      'best_translator.pth')

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

