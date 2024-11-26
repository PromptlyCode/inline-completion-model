import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Step 1: Parsing Python Files
def parse_python_files(directory):
    """
    Parse Python files in a directory to generate input and target texts
    for seq2seq training.
    """
    input_texts = []
    target_texts = []

    # Collect all Python files from the directory
    python_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith(".py")
    ]

    # Process each Python file
    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Pair consecutive lines as input-output for training
            for i in range(len(lines) - 1):
                input_text = lines[i].strip()
                target_text = lines[i + 1].strip()

                # Skip empty lines to maintain meaningful pairs
                if input_text and target_text:
                    input_texts.append(input_text)
                    target_texts.append(target_text)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return input_texts, target_texts

# Directory path
directory_path = "/home/xlisp/EmacsPyPro/jim-emacs-fun-py"

# Parse Python files
input_texts, target_texts = parse_python_files(directory_path)
print(f"Total pairs: {len(input_texts)}")

# Step 2: Dataset and Custom Collate Function
class CodeDataset(Dataset):
    def __init__(self, input_texts, target_texts):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.vocab = self.build_vocab(input_texts, target_texts)

    def build_vocab(self, input_texts, target_texts):
        all_chars = set("".join(input_texts + target_texts))
        return {char: idx for idx, char in enumerate(sorted(all_chars))}

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_seq = torch.tensor([self.vocab[char] for char in self.input_texts[idx]], dtype=torch.long)
        target_seq = torch.tensor([self.vocab[char] for char in self.target_texts[idx]], dtype=torch.long)
        return input_seq, target_seq

def collate_fn(batch):
    """
    Custom collate function to pad input and target sequences in a batch.
    """
    input_seqs, target_seqs = zip(*batch)
    
    # Pad sequences to the same length
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)

    return input_seqs_padded, target_seqs_padded

# Create Dataset and DataLoader
dataset = CodeDataset(input_texts, target_texts)
VOCAB = dataset.vocab
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Step 3: Seq2Seq Model
class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5):
        # Encoder
        embedded = self.embedding(input_seq)
        _, (hidden, cell) = self.encoder(embedded)

        # Decoder
        decoder_input = input_seq[:, 0].unsqueeze(1)  # Start token
        outputs = []
        for t in range(target_seq.size(1)):
            decoder_embedded = self.embedding(decoder_input)
            output, (hidden, cell) = self.decoder(decoder_embedded, (hidden, cell))
            output = self.fc(output)
            outputs.append(output)
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t].unsqueeze(1)  # Teacher forcing
            else:
                decoder_input = output.argmax(2)
        outputs = torch.cat(outputs, dim=1)
        return outputs

# Initialize Model, Loss, and Optimizer
vocab_size = len(VOCAB)
embed_size = 128
hidden_size = 256
model = Seq2SeqModel(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding value in loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training Loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for input_seq, target_seq in dataloader:
        optimizer.zero_grad()

        # Shift target sequence for training
        target_seq_input = target_seq[:, :-1]  # Exclude the last token
        target_seq_output = target_seq[:, 1:]  # Exclude the first token

        # Forward pass
        outputs = model(input_seq, target_seq_input)
        outputs = outputs.permute(0, 2, 1)  # Reshape for loss computation

        # Compute loss
        loss = criterion(outputs, target_seq_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# Step 5: Testing with Prediction
def predict(model, input_seq, max_len=50):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor([[VOCAB[char] for char in input_seq]], dtype=torch.long)
        outputs = model(input_seq)
        predicted = outputs.argmax(2).squeeze(0)
        return ''.join([list(VOCAB.keys())[idx] for idx in predicted])

test_input = "def func(x):"
print("Prediction:", predict(model, test_input))

