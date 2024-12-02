import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

class NumeralTranslationDataset:
    def __init__(self):
        # Mapping of Arabic numerals to English words
        self.num_to_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', 
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', 
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', 
            '17': 'seventeen', '18': 'eighteen', '19': 'nineteen',
            '20': 'twenty', '30': 'thirty', '40': 'forty', '50': 'fifty', 
            '60': 'sixty', '70': 'seventy', '80': 'eighty', '90': 'ninety'
        }

    def generate_training_data(self, num_samples=1000):
        """Generate training data for number translation."""
        input_sequences = []
        target_sequences = []

        # Generate numbers from 0 to 99
        for num in range(100):
            # Convert number to string
            num_str = str(num)
            
            # Translate to words
            if num in self.num_to_words:
                word = self.num_to_words[num_str]
            elif num < 20:
                # Handle teens
                units = str(num % 10)
                word = self.num_to_words[units]
            else:
                # Handle 21-99
                tens = str((num // 10) * 10)
                units = str(num % 10)
                tens_word = self.num_to_words[tens]
                units_word = self.num_to_words[units] if units != '0' else ''
                word = f"{tens_word} {units_word}".strip()
            
            input_sequences.append(list(num_str))
            target_sequences.append(list(word))

        return input_sequences, target_sequences

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded, hidden)
        predictions = self.fc(outputs)
        return predictions, hidden

class Seq2SeqTranslator(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim):
        super(Seq2SeqTranslator, self).__init__()
        self.encoder = Encoder(input_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(output_size, embedding_dim, hidden_dim)
        
    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(0)
        target_len = target_seq.size(1)
        target_vocab_size = self.decoder.fc.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size)
        
        # Encoder
        encoder_outputs, hidden = self.encoder(input_seq)
        
        # First decoder input
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long)
        
        # Decode
        for t in range(target_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t:t+1, :] = decoder_output
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(2)
            
            if teacher_force:
                decoder_input = target_seq[:, t:t+1]
            else:
                decoder_input = top1
        
        return outputs

class NumeralTranslator:
    def __init__(self, input_chars, output_chars):
        self.dataset = NumeralTranslationDataset()
        
        # Create character to index mappings
        self.input_char_to_idx = {char: i for i, char in enumerate(input_chars)}
        self.input_idx_to_char = {i: char for char, i in self.input_char_to_idx.items()}
        
        self.output_char_to_idx = {char: i for i, char in enumerate(output_chars)}
        self.output_idx_to_char = {i: char for char, i in self.output_char_to_idx.items()}
        
        # Hyperparameters
        self.embedding_dim = 64
        self.hidden_dim = 128
        
        # Initialize model
        self.model = Seq2SeqTranslator(
            input_size=len(input_chars),
            output_size=len(output_chars),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
    
    def prepare_sequence(self, seq, char_to_idx):
        """Convert sequence of characters to tensor of indices."""
        return torch.tensor([char_to_idx.get(char, 0) for char in seq], dtype=torch.long)
    
    def pad_sequences(self, sequences, pad_token):
        """Pad sequences to equal length."""
        # Convert sequences to lists if they are tensors
        sequences = [seq.tolist() if torch.is_tensor(seq) else seq for seq in sequences]
        
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            padded.append(seq + [pad_token] * (max_len - len(seq)))
        return torch.tensor(padded, dtype=torch.long)
    
    def train(self, epochs=100, batch_size=32):
        """Train the translation model."""
        # Generate training data
        input_sequences, target_sequences = self.dataset.generate_training_data()
        
        # Prepare input and target sequences
        input_chars = [list(str(seq)) for seq in input_sequences]
        target_chars = [list(seq) for seq in target_sequences]
        
        # Training loop
        epoch_losses = []
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle data
            combined = list(zip(input_chars, target_chars))
            random.shuffle(combined)
            input_chars, target_chars = zip(*combined)
            
            for i in range(0, len(input_chars), batch_size):
                batch_input = input_chars[i:i+batch_size]
                batch_target = target_chars[i:i+batch_size]
                
                # Prepare input sequences
                input_seqs = self.pad_sequences(
                    [self.prepare_sequence(seq, self.input_char_to_idx) for seq in batch_input], 
                    pad_token=0
                )
                
                # Prepare target sequences
                target_seqs = self.pad_sequences(
                    [self.prepare_sequence(seq, self.output_char_to_idx) for seq in batch_target], 
                    pad_token=0
                )
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_seqs, target_seqs)
                
                # Compute loss
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), 
                    target_seqs.view(-1)
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Record average epoch loss
            avg_loss = total_loss / (len(input_chars) // batch_size)
            epoch_losses.append(avg_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        
        # Visualize training loss
        self.plot_training_loss(epoch_losses)
        
        return epoch_losses
    
    def translate(self, input_number):
        """Translate a single number to words."""
        # Prepare input sequence
        input_seq = self.prepare_sequence(list(str(input_number)), self.input_char_to_idx)
        input_seq = input_seq.unsqueeze(0)  # Add batch dimension
        
        # Create dummy target sequence of zeros
        dummy_target = torch.zeros_like(input_seq)
        
        # Disable gradient computation
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_seq, dummy_target)
            
            # Get the most likely output characters
            predicted_indices = outputs.argmax(dim=-1)
            
            # Ensure we have the correct tensor shape
            if predicted_indices.dim() > 2:
                predicted_indices = predicted_indices.squeeze(0)
            
            # Convert indices back to characters
            predicted_chars = [self.output_idx_to_char[idx.item()] for idx in predicted_indices[0]]
            
            # Join characters to form a word
            return ''.join(predicted_chars).strip()
    
    
    def save_model(self, filepath='numeral_translator.pth'):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_char_to_idx': self.input_char_to_idx,
            'output_char_to_idx': self.output_char_to_idx
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='numeral_translator.pth'):
        """Load model state."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.input_char_to_idx = checkpoint['input_char_to_idx']
        self.output_char_to_idx = checkpoint['output_char_to_idx']
        print(f"Model loaded from {filepath}")
    
    def plot_training_loss(self, losses):
        """Visualize training loss."""
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_loss.png')
        plt.close()


def main():
    # Define character sets
    input_chars = ['<pxad>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    output_chars = ['<pxad>', '<bxos>', '<exos>',
                    'z', 'e', 'r', 'o', 'n', 't', 'w', 'h', 'i',
                    'f', 'u', 's', 'v', 'g', 'l', 'y', 'a', 'x']

    # Create translator
    translator = NumeralTranslator(input_chars, output_chars)

    # Train the model
    losses = translator.train(epochs=200)

    # Save the model
    translator.save_model()

    # Demonstrate translation
    test_numbers = [0, 1, 12, 23, 45, 67, 89]
    for num in test_numbers:
        translation = translator.translate(num)
        print(f"{num}: {translation}")

    return translator

if __name__ == '__main__':
    translator = main()
