import torch, pickle, pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from attention_transformer import tokenize, sentence_to_indices, TransformerParaphraser, MAX_LEN, DEVICE, load_glove_embeddings

def load_model(input_vocab, target_vocab, input_emb, target_emb):
    model = TransformerParaphraser(input_vocab, target_vocab, input_emb, target_emb).to(DEVICE)
    model.load_state_dict(torch.load("paraphrase_model.pt", map_location=DEVICE))
    model.eval()
    return model

def beam_decode(model, sentence, input_vocab, target_vocab, idx2word, beam_width=5, length_penalty=0.6):
    src = torch.tensor(sentence_to_indices(sentence, input_vocab)[:MAX_LEN])
    src = src.unsqueeze(0).to(DEVICE)
    memory = model.src_embed(src) + model.pos_enc[:, :src.size(1)]
    memory = model.tr.encoder(memory)
    beams = [(torch.tensor([target_vocab['<sos>']], device=DEVICE), 0.0, [])]

    for _ in range(MAX_LEN):
        new_beams = []
        for prefix, score, words in beams:
            if words and words[-1] == "<eos>":
                new_beams.append((prefix, score, words))
                continue
            tgt_embed = model.tgt_embed(prefix.unsqueeze(0)) + model.pos_enc[:, :prefix.size(0)]
            tgt_mask = model.generate_square_subsequent_mask(prefix.size(0)).to(DEVICE)
            output = model.tr.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            logits = model.fc(output[:, -1])
            probs = torch.log_softmax(logits, dim=1)
            topk = torch.topk(probs, beam_width, dim=1)
            for i in range(beam_width):
                idx = topk.indices[0, i].item()
                token = idx2word.get(idx, "<unk>")
                new_score = score + topk.values[0, i].item()
                new_beams.append((
                    torch.cat([prefix, torch.tensor([idx], device=DEVICE)]),
                    new_score,
                    words + [token]
                ))
        beams = sorted(new_beams, key=lambda x: x[1] / ((len(x[2]) + 1) ** length_penalty), reverse=True)[:beam_width]

    beams = [b for b in beams if len(b[2]) > 0]
    return beams[0][2] if beams else ["<unk>"]

def main():
    print("Loading vocab and model...")
    with open("vocab.pkl", "rb") as f:
        input_vocab, target_vocab = pickle.load(f)
    input_emb = load_glove_embeddings(input_vocab)
    target_emb = load_glove_embeddings(target_vocab)
    model = load_model(input_vocab, target_vocab, input_emb, target_emb)
    idx2word = {i: w for w, i in target_vocab.items()}

    print("Loading test data...")
    df = pd.read_csv("quora-question-pairs/test.csv").dropna()
    inputs = df["question1"].tolist()[:100]
    references = df["question2"].tolist()[:100]

    smoothie = SmoothingFunction().method4
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []

    for i, (inp, ref) in enumerate(zip(inputs, references)):
        pred_tokens = beam_decode(model, inp, input_vocab, target_vocab, idx2word)
        ref_tokens = tokenize(ref)
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        rouge = scorer.score(" ".join(ref_tokens), " ".join(pred_tokens))["rougeL"].fmeasure
        results.append({
            "ID": i+1,
            "Input": inp,
            "Output": " ".join(pred_tokens).replace("<eos>", ""),
            "Target": ref,
            "BLEU": round(bleu, 4),
            "ROUGE-L": round(rouge, 4)
        })

    pd.DataFrame(results).to_csv("paraphrase_results.csv", index=False)
    print("✅ Saved results to paraphrase_results.csv")

if __name__ == "__main__":
    main()
