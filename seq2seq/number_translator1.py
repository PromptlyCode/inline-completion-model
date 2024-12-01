# To implement a model for translating Arabic numerals (like `1234`) into English words (like "one two three four") using a seq2seq approach in PyTorch, we'll break down the task into several parts:

# 1. **Generate Training Data**
# 2. **Build the Seq2Seq Model**
# 3. **Train the Model**
# 4. **Save and Load Model Functions**
# 5. **Visualize the Process**

# ### 1. Generate Training Data

# First, we need to generate a dataset that pairs Arabic numerals with their English equivalents. For simplicity, let's generate a small dataset.

# ```python
import random

def generate_data(num_examples=10000, max_num_length=5):
    numerals = [str(i) for i in range(10)]  # Single digits '0' to '9'
    data = []

    for _ in range(num_examples):
        num_length = random.randint(1, max_num_length)  # Random length of numeral
        number = ''.join(random.choices(numerals, k=num_length))
        english_words = ' '.join([str(int(digit)) for digit in number])  # Mapping digits to words

        data.append((number, english_words))

    return data

train_data = generate_data()
# ```

# ### 2. Build the Seq2Seq Model

# Now, let's implement the seq2seq model. We will use an encoder-decoder architecture with LSTM units.

# #### Encoder

# The encoder will process the input Arabic numeral sequence and encode it into a fixed-length context vector.

# #### Decoder

# The decoder will generate the English sequence one word at a time based on the context vector.

# Here’s the implementation of the encoder-decoder seq2seq model:

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class Seq2SeqDataset(Dataset):
    def __init__(self, data, input_vocab, target_vocab):
        self.data = data
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        number, words = self.data[idx]
        input_seq = [self.input_vocab[char] for char in number]
        target_seq = [self.target_vocab[word] for word in words.split()]
        return torch.tensor(input_seq), torch.tensor(target_seq)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        hidden, cell = self.encoder(input_seq)

        output_seq = []
        decoder_input = target_seq[:, 0].unsqueeze(1)  # First token for the decoder

        for t in range(1, target_seq.size(1)):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            top1 = output.argmax(2)
            output_seq.append(top1)
            decoder_input = top1  # Use current prediction as next input

        return torch.cat(output_seq, dim=1)
# ```

# ### 3. Training the Model

# Now, let's train the model.

# ```python
# Hyperparameters
input_vocab = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '<pad>': 10}
target_vocab = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '<pad>': 10}
input_size = len(input_vocab)
output_size = len(target_vocab)
hidden_size = 256
batch_size = 64
num_epochs = 10

# Dataset and DataLoader
dataset = Seq2SeqDataset(train_data, input_vocab, target_vocab)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
encoder = Encoder(input_size, hidden_size)
decoder = Decoder(output_size, hidden_size)
model = Seq2Seq(encoder, decoder)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=input_vocab['<pad>'])
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_seq, target_seq in dataloader:
        optimizer.zero_grad()

        output_seq = model(input_seq, target_seq)

        # Flatten the sequences
        output_seq = output_seq.view(-1, output_size)
        target_seq = target_seq.view(-1)

        loss = criterion(output_seq, target_seq)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')
# ```

# ### 4. Save and Load Model Functions

# ```python
# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
# ```

# ### 5. Visualizing the Process

# You can visualize the training process by plotting the loss over epochs or using TensorBoard.

# Here’s an example using Matplotlib to plot the training loss:

# ```python
import matplotlib.pyplot as plt

# After training loop, plot the loss
epochs = list(range(1, num_epochs+1))
losses = [total_loss/len(dataloader)]  # Store loss after each epoch

plt.plot(epochs, losses, label="Training Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()
# ```

# ### Example Usage

# ```python
# Save the trained model
save_model(model, "seq2seq_model.pth")

# Load the model for inference
model = Seq2Seq(encoder, decoder)
load_model(model, "seq2seq_model.pth")
# ```

# This is a simplified version of the sequence-to-sequence model for translating Arabic numerals into English. You can refine the training loop, add validation, and improve the model architecture based on your requirements.
