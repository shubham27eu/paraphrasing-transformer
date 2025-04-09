
⸻



# ✨ Transformer-Based Paraphrasing System

This project implements a custom Transformer encoder-decoder model trained on the Quora Question Pairs dataset to generate fluent paraphrases. It includes enhancements like:

- ✅ Beam search decoding with length penalty  
- ✅ N-gram blocking to reduce redundancy  
- ✅ `<unk>` token replacement using attention weights  
- ✅ BLEU and ROUGE-L evaluation tracking  
- ✅ No LLMs used — this is a fully custom model from scratch

---

## 📂 Project Structure

nlp_project/
├── attention_transformer.py        # Transformer model + training logic
├── run_inference.py                # Beam search decoding +  replacement
├── evaluate_and_export.py          # BLEU + ROUGE-L scoring, CSV export
├── glove.6B.100d.txt               # ⚠️ Must be downloaded manually (see below)
├── vocab.pkl                       # Pickled vocabularies
├── paraphrase_model.pt             # Trained PyTorch model weights
├── .gitignore

---

## 🧠 Dataset

We use the [Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs). Only entries marked with `is_duplicate == 1` are used during training to ensure the model learns to paraphrase rather than detect duplicates.

---

## 📥 Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/shubham27eu/paraphrasing-transformer.git
cd paraphrasing-transformer

2. Create a virtual environment and install dependencies

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

If requirements.txt doesn’t exist, use this:

torch
numpy
pandas
nltk
scikit-learn
matplotlib



⸻

📦 Download GloVe Embeddings

This project uses GloVe (100d) word embeddings. Due to GitHub’s file size restriction, it is not included here.

Download GloVe from https://nlp.stanford.edu/data/glove.6B.zip, unzip it, and place the glove.6B.100d.txt file into your project folder:

mv glove.6B.100d.txt /path/to/paraphrasing-transformer/



⸻

🚀 How to Run

Train the model (Optional if paraphrase_model.pt is available):

python attention_transformer.py

Run inference on test questions:

python run_inference.py

Evaluate and export BLEU and ROUGE-L scores:

python evaluate_and_export.py



⸻

📊 Example Output

Input: How can I reduce belly fat?
Output: what is the best way to reduce belly fat ?

Input: Should I get a hair transplant at 24?
Output: what would be the cost of a hair transplant ?



⸻

🔍 Key Features
	•	✅ Custom Transformer encoder-decoder model (no pre-trained LLMs)
	•	✅ Beam search decoding with length penalties
	•	✅ Attention-based <unk> token resolution
	•	✅ Bigram-level n-gram blocking for fluent output
	•	✅ BLEU + ROUGE-L evaluation
	•	✅ Human-readable paraphrase output
	•	✅ Dataset-limited to real Quora duplicates for training

⸻

🚫 No LLMs Used

This project does not use any Large Language Models (LLMs) such as BERT, GPT, or T5. It is built from scratch using a Transformer and trained end-to-end on real-world data.
