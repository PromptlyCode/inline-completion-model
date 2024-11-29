import os
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
from  dataloader import CodeDataset, parse_python_files, collate_fn
from seq2seq_model import Seq2SeqModel

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

directory_path = "/home/xlisp/Desktop/code_dataset"

# Parse Python files
input_texts, target_texts = parse_python_files(directory_path)
print(f"Total pairs: {len(input_texts)}")

# Create Dataset and DataLoader
dataset = CodeDataset(input_texts, target_texts)
VOCAB = dataset.vocab
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Initialize Model, Loss, and Optimizer
vocab_size = len(VOCAB)
embed_size = 128
hidden_size = 256
model = Seq2SeqModel(vocab_size, embed_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding value in loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Training Loop
epochs = 300
for epoch in range(epochs):
    total_loss = 0
    for input_seq, target_seq in dataloader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)  # Move data to GPU
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

# Step 5: Save Model and FAISS Index
# Save the model weights
torch.save(model.state_dict(), "seq2seq_model.pth")

# Saving embeddings (encoder hidden states) to FAISS
# Assume we want to index the hidden states from the encoder:
all_embeddings = []
model.eval()
with torch.no_grad():
    for input_seq, target_seq in dataloader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        embedded = model.embedding(input_seq)  # Get embedded inputs
        _, (hidden, _) = model.encoder(embedded)
        all_embeddings.append(hidden[-1].cpu().numpy())  # Take the last hidden state

# Convert to numpy array and build FAISS index
embeddings = np.concatenate(all_embeddings, axis=0).astype('float32')
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance index
index.add(embeddings)  # Add embeddings to FAISS

# Save the FAISS index to a file
faiss.write_index(index, "embeddings.index")

# Step 6: Load Model and FAISS Index
# Load the model
model.load_state_dict(torch.load("seq2seq_model.pth"))
model.to(device)

# Load the FAISS index
index = faiss.read_index("embeddings.index")

# Step 7: Testing with Prediction
def predict(model, input_seq, max_len=50):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor([[VOCAB[char] for char in input_seq]], dtype=torch.long).to(device)
        outputs = model(input_seq, target_seq=None)  # Pass target_seq as None
        predicted = outputs.argmax(2).squeeze(0).cpu().numpy()
        return ''.join([list(VOCAB.keys())[idx] for idx in predicted])

test_input = "public long get" #"def func" #(x):"
print("Prediction:", predict(model, test_input))

