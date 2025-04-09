import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch, pickle, nltk
import pandas as pd
import numpy as np
from attention_transformer import tokenize, sentence_to_indices, TransformerParaphraser, MAX_LEN, DEVICE, load_glove_embeddings

nltk.download("punkt")

def load_model(input_vocab, target_vocab, input_emb, target_emb):
    model = TransformerParaphraser(input_vocab, target_vocab, input_emb, target_emb).to(DEVICE)
    model.load_state_dict(torch.load("paraphrase_model.pt", map_location=DEVICE))
    model.eval()
    return model

def beam_decode(model, sentence, input_vocab, target_vocab, idx2word, beam_width=5, max_len=MAX_LEN, alpha=0.7):
    model.eval()
    src_tokens = sentence_to_indices(sentence, input_vocab)
    src_tensor = torch.tensor(src_tokens[:MAX_LEN] + [input_vocab['<pad>']] * (MAX_LEN - len(src_tokens))).unsqueeze(0).to(DEVICE)

    src_mask = (src_tensor == input_vocab['<pad>'])
    memory = model.src_embed(src_tensor) + model.pos_enc[:, :src_tensor.size(1)].to(DEVICE)
    memory = model.tr.encoder(memory, src_key_padding_mask=src_mask)

    beams = [(torch.tensor([target_vocab['<sos>']], device=DEVICE), [], 0.0)]  # (tokens, words, score)

    for _ in range(max_len):
        new_beams = []
        for tokens, words, score in beams:
            tgt_mask = model.generate_square_subsequent_mask(tokens.size(0)).to(DEVICE)
            tgt_emb = model.tgt_embed(tokens.unsqueeze(0)) + model.pos_enc[:, :tokens.size(0)]
            decoder_out = model.tr.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

            logits = model.fc(decoder_out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)
            topk = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                idx = topk.indices[0][i].item()
                word = idx2word[idx]
                new_score = score + topk.values[0][i].item()

                # N-gram blocking (bigrams)
                if len(words) >= 1 and f"{words[-1]} {word}" in {" ".join(words[i:i+2]) for i in range(len(words)-1)}:
                    continue

                if word == "<eos>":
                    return " ".join(words)

                # UNK replacement using attention weights
                if word == "<unk>":
                    attn_weights = decoder_out[:, -1].detach().cpu().numpy()
                    if len(src_tokens) > 0:
                        max_idx = min(np.argmax(attn_weights), len(src_tokens) - 1)
                        src_token_idx = src_tokens[max_idx]
                        word = next((w for w, i in input_vocab.items() if i == src_token_idx), "<unk>")

                new_tokens = torch.cat([tokens, torch.tensor([idx], device=DEVICE)])
                new_beams.append((new_tokens, words + [word], new_score))

        # Apply length penalty and keep top beams
        beams = sorted(new_beams, key=lambda x: x[2] / ((5 + len(x[1])) ** alpha / 6**alpha), reverse=True)[:beam_width]

    return " ".join(beams[0][1]) if beams else "<unk>"

def main():
    print("Loading vocabularies and model...")
    with open("vocab.pkl", "rb") as f:
        input_vocab, target_vocab = pickle.load(f)
    input_emb = load_glove_embeddings(input_vocab)
    target_emb = load_glove_embeddings(target_vocab)
    model = load_model(input_vocab, target_vocab, input_emb, target_emb)
    idx2word = {i: w for w, i in target_vocab.items()}

    print("Loading test data...")
    df = pd.read_csv("quora-question-pairs/test.csv").dropna()
    test_sentences = df["question1"].tolist()[:10]

    print("\nGenerating paraphrases (beam decoding):\n")
    for i, s in enumerate(test_sentences):
        output = beam_decode(model, s, input_vocab, target_vocab, idx2word)
        print(f"[{i+1}] Input: {s}\n     Output: {output}\n")

if __name__ == "__main__":
    main()
