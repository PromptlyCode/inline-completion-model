from torch.utils.data import Dataset, DataLoader

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
                lines = [line for line in lines if not re.match(r'^\s*#', line) and not re.search(r'#', line)]

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

