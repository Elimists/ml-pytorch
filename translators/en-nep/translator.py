print("Importing modules...")
try:
    import pandas as pd
    import spacy
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random
    from torch.utils.data import Dataset, DataLoader
    import ast
    from torch.nn.utils.rnn import pad_sequence
    import concurrent.futures
    from translators.utils.Encoder import Encoder
    from translators.utils.Decoder import Decoder
    from translators.utils.Seq2Seq import Seq2Seq

    nlp = spacy.load('en_core_web_sm')
    print("All modules imported successfully! \n")
except ImportError as e:
    print(f"An import error occurred: {e}. Please install the missing module and try again.")
    exit()
except OSError:
    print("The 'en_core_web_sm' model is not installed. Please install it before running this script.")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = spacy.load('en_core_web_sm')

# Read the .parquet data file containing the English and Nepali text
data = pd.read_parquet('en-ne.parquet')
print("data loaded successfully...")

# Process the text data, convert it to lowercase, and remove punctuation.
counter = {'en': 0, 'ne': 0}
def preprocess_text(text, column):
    # Lowercase the text
    text = text.lower()

    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha]

    

    # Print a message after every 5000 applications
    if tokens:
        counter[column] += 1
        if counter[column] % 5000 == 0:
             print(f"Processed {counter[column]} texts in {column} column")

    return tokens


def preprocess_column(data, column):
    print(f"Processing {column} column...")
    return data[column].apply(lambda text: preprocess_text(text, column))

"""
# Apply the preprocess_text function to the 'en' and 'ne' columns. Use ThreadPoolExecutor to speed up the process by running the two functions concurrently.
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_en = executor.submit(preprocess_column, data, 'en')
    future_ne = executor.submit(preprocess_column, data, 'ne')

    data['en'] = future_en.result()
    data['ne'] = future_ne.result()

# Save the DataFrame to a CSV file
data.to_csv('preprocessed_data.csv', index=False)
"""
# read the preprocessed data from the CSV file
data = pd.read_csv('preprocessed_data.csv')

# Convert the strings in the 'en' and 'ne' columns back into lists
data['en'] = data['en'].apply(ast.literal_eval)
data['ne'] = data['ne'].apply(ast.literal_eval)


# Create a set of unique words in English and Nepali
print("creating unique vocabularies...")
vocab_en = set(word for tokens in data['en'] for word in tokens)
vocab_ne = set(word for tokens in data['ne'] for word in tokens)
print("unique vocabularies created successfully! \n")

vocab_ne.add('<pad>')

# Create word to index mapping
print("creating word to index mapping...")
word2idx_en = {word: idx for idx, word in enumerate(vocab_en)}
word2idx_ne = {word: idx for idx, word in enumerate(vocab_ne)}
print("word to index mapping created successfully! \n")

# Convert words to integers
print("converting words to integers...")
data['en'] = data['en'].apply(lambda tokens: [word2idx_en[word] for word in tokens])
data['ne'] = data['ne'].apply(lambda tokens: [word2idx_ne[word] for word in tokens])
print("words converted to integers successfully! \n")

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

# Create instances of the dataset
dataset = TextDataset(data)

# Define a function to collate data samples into batches
def collate_fn(batch):
    src = [item['en'] for item in batch]
    trg = [item['ne'] for item in batch]
    return {'src': src, 'trg': trg}

BATCH_SIZE = 64
iterator = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Initialize the model
print("initializing the model...")
INPUT_DIM = len(word2idx_en)
OUTPUT_DIM = len(word2idx_ne)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
TRG_PAD_IDX = word2idx_ne['<pad>']
N_EPOCHS = 10

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
print("model initialized successfully! \n")

# Define the loss function and optimizer
print("defining the loss function and optimizer...")
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
optimizer = optim.Adam(model.parameters())
print("loss function and optimizer defined successfully! \n")



# Training loop
print("training the model...")
for epoch in range(N_EPOCHS):
    
    model.train()

    # Set the model to training mode
    for param in model.parameters():
        param.requires_grad = True
    
    # Iterate over the training data
    for i, batch in enumerate(iterator):
        
        src = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in batch['src']])
        trg = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in batch['trg']])
    
        optimizer.zero_grad()
    
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        optimizer.step()
    
    # Save the model's state
    torch.save(model.state_dict(), f'model_{epoch}.pt')
print("model trained successfully! \n")
print("model saved successfully as 'model.pt'! \n")