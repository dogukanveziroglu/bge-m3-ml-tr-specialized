---
license: apache-2.0
datasets:
- dogukanvzr/ml-paraphrase-tr
language:
- tr
base_model:
- BAAI/bge-m3
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- embedding
- paraphrase-identification
- semantic-search
- text-embedding
- dense
- turkish
- technical-language
- scientific-text
- huggingface
- transformer
- multilingual
- cosine-similarity
- ml-paraphrase-tr
library_name: sentence-transformers
---
# bge-m3-ml-tr-specialized

`bge-m3-ml-tr-specialized` is a Sentence Transformer model optimized for scientific and technical machine learning texts in Turkish, specifically in the field of machine learning. Based on `BAAI/bge-m3`, the model has been fine-tuned for tasks such as sentence similarity, semantic search, conceptual matching, and meaning-based classification.

## 🧠 Model Specifications

- **Model Type:** Sentence Transformer  
- **Base Model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)  
- **Use Cases:**
  - Sentence-level semantic similarity
  - Conceptual and contextual sentence alignment
  - Information retrieval and semantic search systems
  - Clustering and ranking of scientific documents
- **Language:** Turkish (especially technical and scientific domain)
- **Maximum Sequence Length:** 8192 tokens  
- **Output Vector Dimension:** 1024  
- **Pooling Strategy:** CLS token  
- **Similarity Metric:** Cosine Similarity

## 🔍 Model Architecture

```python
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'architecture': 'XLMRobertaModel'})
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True})
  (2): Normalize()
)
```

## 🚀 Quick Start

```bash
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("dogukanvzr/bge-m3-ml-tr-specialized")

sentences = [
    "Accuracy refers to how close a model's predictions are to the actual values.",
    "Model accuracy indicates how well the predictions align with true labels.",
    "Feature engineering plays a critical role in machine learning pipelines."
]

embeddings = model.encode(sentences)

from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity([embeddings[0]], embeddings[1:])
print(scores)
```

## 🧪 Training Details

- **Dataset:** [`ml-paraphrase-tr`](https://huggingface.co/datasets/dogukanvzr/ml-paraphrase-tr)  
- **Size:** 60,000 sentence pairs  
- **Structure:** `sentence_0`, `sentence_1`, `label` (float between 0.0–1.0 indicating similarity)  
- **Loss Function:** `CosineSimilarityLoss` (internally uses `MSELoss`)  
- **Training Epochs:** 3  
- **Batch Size:** 64  

### 📈 Training Log

| Epoch | Step | Average Loss |
|-------|------|---------------|
| 0.5   | 500  | 0.0338        |
| 1.0   | 1000 | 0.0188        |
| 1.5   | 1500 | 0.0147        |
| 2.0   | 2000 | 0.0127        |
| 2.5   | 2500 | 0.0105        |

## 📊 Application Areas

This model is particularly well-suited for the following NLP and ML tasks in Turkish:

- **Semantic alignment** in technical documents  
- **Similarity detection** in scientific and academic texts  
- **Embedding-based information retrieval**  
- **Paraphrase identification** (detecting meaning-equivalent sentence pairs)  
- **Semantic clustering** for topic grouping  
- **Intent matching** in QA and chatbot systems

## 💡 Evaluation Example

```python
s1 = "Machine learning algorithms learn from past data to make future predictions."
s2 = "The model performs inference based on learned patterns."
s3 = "The size of the dataset can affect the generalization capacity of the model."

embs = model.encode([s1, s2, s3])
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embs[0]], embs[1:])
print(sim)
```

## ⚙️ Development Environment

- Python: 3.12.7  
- Sentence Transformers: 5.0.0  
- Transformers: 4.56.0.dev0  
- PyTorch: 2.7.1+cu128  
- Accelerate: 1.9.0  
- Datasets: 4.0.0  
- Tokenizers: 0.21.4  

## 📚 Citation

```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

## ⚠️ Limitations

- The model is trained primarily on scientific/technical Turkish text and may underperform on casual, figurative, or conversational language.
- It might struggle with cultural references, idioms, or sarcasm.
- Although trained on high-quality paraphrased data, users should still review outputs critically.

## 📬 Contact & Feedback

For bug reports, suggestions, or contributions:

- 📧 Hugging Face Profile: [@dogukanvzr](https://huggingface.co/dogukanvzr)  
- 📂 Dataset used for training: [`ml-paraphrase-tr`](https://huggingface.co/datasets/dogukanvzr/ml-paraphrase-tr)