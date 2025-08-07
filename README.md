---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:60000
- loss:CosineSimilarityLoss
base_model: BAAI/bge-m3
widget:
- source_sentence: Yüksek doğruluk, bir modelin tahminlerinin gerçek değerlere yakın
    olması anlamına gelir.
  sentences:
  - Bir modelin tahminlerinin gerçek değerlere yakınlığı, yüksek doğruluk seviyesini
    gösterir.
  - Görüntü işlemede, pooling katmanları boyut azaltma sağlayarak hesaplama verimliliğini
    artırır.
  - Doğal Dil İşleme sistemlerinin geliştirilmesinde, makine öğrenmesi algoritmaları
    kullanılır.
- source_sentence: Yapay sinir ağları, verileri işleyerek tahminlerde bulunma ve sınıflandırma
    yapma gibi görevleri yerine getirirler.
  sentences:
  - Makine öğrenmesi, süpervizyon, öz-süpervizyon ve açık öğrenme gibi çeşitli öğrenme
    yöntemlerine dayanır.
  - t-SNE, veri kümesindeki benzerlikleri koruyarak veri kümesini azaltmak için manifold
    kavramına dayanır.
  - Veri kümesi büyüklüğüne göre, çapraz doğrulamada değişik stratejiler kullanılabilir.
- source_sentence: Temel bileşenler, orijinal değişkenlerin lineer kombinasyonlarıdır.
  sentences:
  - Dikkat, beyindeki bilgi işleme sürecini yönlendiren bir filtre görevi görür.
  - Fonksiyonun minimum değerini bulmak için, ağırlıkların küçük adımlarla ayarlanması
    prensibine göre çalışan gradyan inişi algoritması kullanılır.
  - Veri kümesindeki temel bileşenler, orijinal değişkenlerin ağırlıklandırılmış toplamlarından
    oluşur.
- source_sentence: Aşırı öğrenme, modelin eğitim verilerini aşırı derecede öğrenmesi
    ve yeni, görülmemiş veriler için yeterince iyi genelleme yapamaması durumudur.
  sentences:
  - XGBoost, kullanıcıların model oluşturma sürecini kolaylaştırmak için çeşitli özelliklere
    sahiptir.
  - Makine öğrenmesi algoritmaları, verilerdeki kalıpları analiz ederek gelecekteki
    olayları tahmin etmeye yöneliktir.
  - Aşırı öğrenme, modelin eğitim verilerine aşırı bağımlılığının, yeni verilerle
    uyum sağlayamamasına sebep olur.
- source_sentence: AUC skoru, 0 ile 1 arasında değişen bir değere sahiptir ve 1, mükemmel
    bir sınıflandırma performansı gösterir.
  sentences:
  - Orijinal verileri geri üretmek için girdi verilerini düşük boyutlu temsil haline
    getiren temel amacı olan autoencoderlar, sıkıştırma ve geri dönüştürme süreçlerini
    kullanır.
  - Doğru tahminlerin oranı yüksek olduğunda modelin doğruluğu yüksek olur.
  - AUC skoru, 0'dan 1'e kadar bir aralıkta değişir; 1 değeri, mükemmel bir sınıflandırma
    modeli içindir.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-m3

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) <!-- at revision 5617a9f61b028005a4858fdac845db406aefb181 -->
- **Maximum Sequence Length:** 8192 tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False, 'architecture': 'XLMRobertaModel'})
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("dogukanvzr/ml-paraphrase-tr")
# Run inference
sentences = [
    'AUC skoru, 0 ile 1 arasında değişen bir değere sahiptir ve 1, mükemmel bir sınıflandırma performansı gösterir.',
    "AUC skoru, 0'dan 1'e kadar bir aralıkta değişir; 1 değeri, mükemmel bir sınıflandırma modeli içindir.",
    'Orijinal verileri geri üretmek için girdi verilerini düşük boyutlu temsil haline getiren temel amacı olan autoencoderlar, sıkıştırma ve geri dönüştürme süreçlerini kullanır.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9994, 0.0193],
#         [0.9994, 1.0000, 0.0206],
#         [0.0193, 0.0206, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 60,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                          |
  | details | <ul><li>min: 5 tokens</li><li>mean: 25.49 tokens</li><li>max: 50 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 24.73 tokens</li><li>max: 43 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.75</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                         | sentence_1                                                                                               | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Bu ağlar, geçmiş girdileri saklamak için gizli durumlar kullanır, böylece zaman içindeki verilerin doğal ardışık yapısını yakalarlar.</code> | <code>Güvenilir ve tutarlı tahminler yüksek kesinlik ile ilişkilidir.</code>                             | <code>0.0</code> |
  | <code>Bu alanda kullanılan teknikler arasında dil modellemesi ve doğal dil anlama yer alır.</code>                                                 | <code>Doğal Dil İşleme, dil modellemesi ve doğal dil anlama gibi teknikleri kullanır.</code>             | <code>1.0</code> |
  | <code>Bu algoritma, her düğümde olası karar noktalarını temsil eden dallara ayrılır ve her dal, belirli bir sonuca götürür.</code>                 | <code>Karar ağaçları, her karar noktasını temsil eden dallar aracılığıyla olası sonuçlara ulaşır.</code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.5330 | 500  | 0.0338        |
| 1.0661 | 1000 | 0.0188        |
| 1.5991 | 1500 | 0.0147        |
| 2.1322 | 2000 | 0.0127        |
| 2.6652 | 2500 | 0.0105        |


### Framework Versions
- Python: 3.12.7
- Sentence Transformers: 5.0.0
- Transformers: 4.56.0.dev0
- PyTorch: 2.7.1+cu128
- Accelerate: 1.9.0
- Datasets: 4.0.0
- Tokenizers: 0.21.4

## Citation

### BibTeX

#### Sentence Transformers
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

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->