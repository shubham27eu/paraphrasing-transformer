import torch, nltk, pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd, numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nltk.download('punkt')
MAX_LEN, EMBED_DIM, HIDDEN_DIM, NUM_HEADS = 20, 100, 256, 4
FF_DIM, NUM_LAYERS, BATCH_SIZE, EPOCHS = 512, 2, 64, 15
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
GLOVE_PATH = "glove.6B.100d.txt"

def tokenize(text): return word_tokenize(text.lower())
def build_vocab(sentences, min_freq=2):
    counter = Counter(token for s in sentences for token in tokenize(s))
    vocab = {'<pad>':0,'<sos>':1,'<eos>':2,'<unk>':3}
    for word, freq in counter.items():
        if freq >= min_freq: vocab[word] = len(vocab)
    return vocab
def load_glove_embeddings(vocab):
    matrix = np.random.uniform(-0.1, 0.1, (len(vocab), EMBED_DIM)); matrix[vocab['<pad>']] = 0
    with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            w, *vec = line.strip().split()
            if w in vocab: matrix[vocab[w]] = np.array(vec, dtype=np.float32)
    return torch.tensor(matrix, dtype=torch.float)

def sentence_to_indices(sentence, vocab):
    return [vocab.get(w, vocab['<unk>']) for w in tokenize(sentence)]

class ParaphraseDataset(Dataset):
    def __init__(self, df, ivocab, tvocab):
        self.inputs = df['question1'].tolist(); self.targets = df['question2'].tolist()
        self.iv, self.tv = ivocab, tvocab
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx):
        src = sentence_to_indices(self.inputs[idx], self.iv)[:MAX_LEN]
        trg = [self.tv['<sos>']] + sentence_to_indices(self.targets[idx], self.tv)[:MAX_LEN-2] + [self.tv['<eos>']]
        return torch.tensor(src + [self.iv['<pad>']]*(MAX_LEN-len(src))), \
               torch.tensor(trg + [self.tv['<pad>']]*(MAX_LEN-len(trg)))

class TransformerParaphraser(nn.Module):
    def __init__(self, input_vocab, target_vocab, emb1, emb2):
        super().__init__()
        self.src_embed = nn.Embedding.from_pretrained(emb1, freeze=False)
        self.tgt_embed = nn.Embedding.from_pretrained(emb2, freeze=False)
        self.pos_enc = nn.Parameter(torch.rand(1, MAX_LEN, EMBED_DIM))
        self.tr = nn.Transformer(d_model=EMBED_DIM, nhead=NUM_HEADS,
                                 num_encoder_layers=NUM_LAYERS, num_decoder_layers=NUM_LAYERS,
                                 dim_feedforward=FF_DIM, batch_first=True)
        self.fc = nn.Linear(EMBED_DIM, len(target_vocab))
        self.pad_idx = target_vocab['<pad>']
    def forward(self, src, tgt):
        src_mask = src == self.pad_idx
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
        src = self.src_embed(src) + self.pos_enc[:, :src.size(1)]
        tgt = self.tgt_embed(tgt) + self.pos_enc[:, :tgt.size(1)]
        out = self.tr(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_mask)
        return self.fc(out)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def train_model(model, loader, opt, criterion, scheduler):
    model.train(); total = 0
    for src, trg in tqdm(loader, desc="Training"):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        opt.zero_grad()
        output = model(src, trg[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step(); total += loss.item()
    scheduler.step()
    return total / len(loader)

def main():
    df = pd.read_csv("quora-question-pairs/train.csv").dropna()
    df = df[df["is_duplicate"] == 1][["question1", "question2"]]
    iv, tv = build_vocab(df['question1']), build_vocab(df['question2'])
    emb1, emb2 = load_glove_embeddings(iv), load_glove_embeddings(tv)
    loader = DataLoader(ParaphraseDataset(df, iv, tv), batch_size=BATCH_SIZE, shuffle=True)
    model = TransformerParaphraser(iv, tv, emb1, emb2).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=tv['<pad>'])

    for epoch in range(EPOCHS):
        loss = train_model(model, loader, opt, criterion, scheduler)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

    torch.save(model.state_dict(), "paraphrase_model.pt")
    with open("vocab.pkl", "wb") as f: pickle.dump((iv, tv), f)

if __name__ == "__main__":
    main()
