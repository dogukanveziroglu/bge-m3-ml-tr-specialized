# bge-m3-ml-tr-specialized

`bge-m3-ml-tr-specialized`, Türkçe bilimsel ve teknik metinler için optimize edilmiş bir Sentence Transformer modelidir. Model, `BAAI/bge-m3` temel alınarak eğitilmiş olup, cümle benzerliği, semantik arama, kavramsal eşleşme ve anlam odaklı sınıflandırma gibi görevlerde kullanılmak üzere tasarlanmıştır.

## 🧠 Model Özellikleri

- **Model Türü:** Sentence Transformer  
- **Taban Model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)  
- **Uygulama Alanları:**
  - Cümle düzeyinde benzerlik hesaplama
  - Semantik bilgiye dayalı metin eşleştirme
  - Bilgi erişimi ve semantik arama sistemleri
  - Bilimsel metinlerin anlamsal kümeleme ve sıralanması
- **Dil:** Türkçe (özellikle teknik ve bilimsel cümleler)
- **Maksimum Girdi Uzunluğu:** 8192 token  
- **Çıktı Vektör Boyutu:** 1024  
- **Havuzlama Yöntemi:** CLS token üzerinden  
- **Benzerlik Ölçütü:** Kosinüs Benzerliği (Cosine Similarity)

## 🔍 Model Mimarisi

\`\`\`python
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'architecture': 'XLMRobertaModel'})
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True})
  (2): Normalize()
)
\`\`\`

## 🚀 Hızlı Başlangıç

\`\`\`bash
pip install -U sentence-transformers
\`\`\`

\`\`\`python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("dogukanvzr/bge-m3-ml-tr-specialized")

sentences = [
    "Doğruluk, bir modelin gerçek değerlere ne kadar yakın sonuçlar verdiğini gösterir.",
    "Model doğruluğu, tahminlerin gerçek etiketlerle örtüşme derecesini yansıtır.",
    "Özellik mühendisliği, makine öğrenmesi süreçlerinde önemli rol oynar."
]

embeddings = model.encode(sentences)

from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity([embeddings[0]], embeddings[1:])
print(scores)
\`\`\`

## 🧪 Eğitim Bilgileri

- **Veri Kümesi:** [`ml-paraphrase-tr`](https://huggingface.co/datasets/dogukanvzr/ml-paraphrase-tr)
- **Boyut:** 60.000 örnek  
- **Yapı:** `sentence_0`, `sentence_1`, `label` (float, 0.0–1.0 arası benzerlik)  
- **Kayıp Fonksiyonu:** `CosineSimilarityLoss` (içsel olarak `MSELoss` kullanılmıştır)  
- **Eğitim Epoch Sayısı:** 3  
- **Batch Size:** 64  

### 📈 Eğitim Süreci

| Epoch | Adım | Ortalama Kayıp (Loss) |
|-------|------|------------------------|
| 0.5   | 500  | 0.0338                 |
| 1.0   | 1000 | 0.0188                 |
| 1.5   | 1500 | 0.0147                 |
| 2.0   | 2000 | 0.0127                 |
| 2.5   | 2500 | 0.0105                 |

## 📊 Kullanım Alanları

Bu model, özellikle aşağıdaki teknik/NLP görevleri için uygundur:

- Türkçe teknik dökümanlarda **anlamsal eşleştirme**
- Akademik ve bilimsel metinlerde **benzerlik analizi**
- **Embedding tabanlı bilgi erişim sistemleri**
- **Paraphrase detection** (anlamca yakın cümle çiftlerinin tespiti)
- **Semantic Clustering** (anlam temelli kümeleme)
- Soru-cevap sistemlerinde **intent eşleştirme**

## 💡 Örnek Değerlendirme

\`\`\`python
s1 = "Makine öğrenmesi algoritmaları, geçmiş verilerden öğrenerek geleceği tahmin eder."
s2 = "Model, öğrenilmiş örüntülerden faydalanarak tahmin yürütür."
s3 = "Veri seti boyutu, modelin genelleme kapasitesini etkileyebilir."

embs = model.encode([s1, s2, s3])
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embs[0]], embs[1:])
print(sim)
\`\`\`

## ⚙️ Geliştirme Ortamı

- Python: 3.12.7  
- Sentence Transformers: 5.0.0  
- Transformers: 4.56.0.dev0  
- PyTorch: 2.7.1+cu128  
- Accelerate: 1.9.0  
- Datasets: 4.0.0  
- Tokenizers: 0.21.4  

## 📚 Atıf

\`\`\`bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
\`\`\`

## ⚠️ Sınırlılıklar

- Model, teknik ve bilimsel dil üzerinde eğitildiği için günlük konuşma dili veya mecaz içeren ifadelerde düşük performans gösterebilir.
- Kültürel bağlam, ironi, deyimsel ifadeler gibi alanlarda genelleme yeteneği sınırlıdır.
- Eğitim verisinde bias oluşmamış olsa da, çıktılar dikkatle değerlendirilmelidir.

## 📬 İletişim ve Geri Bildirim

Model ile ilgili sorun bildirmek, öneride bulunmak ya da katkıda bulunmak için:

- 📧 Hugging Face Profili: [@dogukanvzr](https://huggingface.co/dogukanvzr)
- 📂 Veri kümesi: [`ml-paraphrase-tr`](https://huggingface.co/datasets/dogukanvzr/ml-paraphrase-tr)