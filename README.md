<h1 align="center">PySentence-Similarity 😊</h1>
<p align="center">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/pysentence-similarity">
    <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/pysentence-similarity">
    <a href="https://github.com/goldpulpy/pysentence-similarity/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/goldpulpy/pysentence-similarity.svg?color=blue"></a>
    <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/goldpulpy/pysentence-similarity/package.yml">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/goldpulpy/pysentence-similarity">
</p>

## Information

**PySentence-Similarity** is a tool designed to identify and find similarities between sentences and a base sentence, expressed as a percentage 📊. It compares the semantic value of each input sentence to the base sentence, providing a score that reflects how related or similar they are. This tool is useful for various natural language processing tasks such as clustering similar texts 📚, paraphrase detection 🔍 and textual consequence measurement 📈.

The models were converted to ONNX format to optimize and speed up inference. Converting models to ONNX enables cross-platform compatibility and optimized hardware acceleration, making it more efficient for large-scale or real-world applications 🚀.

- **High accuracy:** Utilizes a robust Transformer-based architecture, providing high accuracy in semantic similarity calculations 🔬.
- **Cross-platform support:** The ONNX format provides seamless integration across platforms, making it easy to deploy across environments 🌐.
- **Scalability:** Efficient processing can handle large datasets, making it suitable for enterprise-level applications 📈.
- **Real-time processing:** Optimized for fast output, it can be used in real-world applications without significant latency ⏱️.
- **Flexible:** Easily adaptable to specific use cases through customization or integration with additional models or features 🛠️.
- **Low resource consumption:** The model is designed to operate efficiently, reducing memory and CPU/GPU requirements, making it ideal for resource-constrained environments ⚡.
- **Fast and user-friendly:** The library offers high performance and an intuitive interface, allowing users to quickly and easily integrate it into their projects 🚀.

## Installation 📦

- **Requirements:** Python 3.8 or higher.

```bash
# install from PyPI
pip install pysentence-similarity

# install from GitHub
pip install git+https://github.com/goldpulpy/pysentence-similarity.git
```

## Support models 🤝

You don't need to download anything; the package itself will download the model and its tokenizer from a special HF [repository](https://huggingface.co/goldpulpy/pysentence-similarity).

Below are the models currently added to the special repository, including their file size and a link to the source.

| Model                                 | Parameters | FP32   | FP16  | INT8  | Source link                                                                                 |
| ------------------------------------- | ---------- | ------ | ----- | ----- | ------------------------------------------------------------------------------------------- |
| paraphrase-albert-small-v2            | 11.7M      | 45MB   | 22MB  | 38MB  | [HF](https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2) 🤗            |
| all-MiniLM-L6-v2                      | 22.7M      | 90MB   | 45MB  | 23MB  | [HF](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 🤗                      |
| paraphrase-MiniLM-L6-v2               | 22.7M      | 90MB   | 45MB  | 23MB  | [HF](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) 🤗               |
| multi-qa-MiniLM-L6-cos-v1             | 22.7M      | 90MB   | 45MB  | 23MB  | [HF](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) 🤗             |
| msmarco-MiniLM-L-6-v3                 | 22.7M      | 90MB   | 45MB  | 23MB  | [HF](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-6-v3) 🤗                 |
| all-MiniLM-L12-v2                     | 33.4M      | 127MB  | 65MB  | 32MB  | [HF](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) 🤗                     |
| gte-small                             | 33.4M      | 127MB  | 65MB  | 32MB  | [HF](https://huggingface.co/thenlper/gte-small) 🤗                                          |
| all-distilroberta-v1                  | 82.1M      | 313MB  | 157MB | 79MB  | [HF](https://huggingface.co/sentence-transformers/all-distilroberta-v1) 🤗                  |
| all-mpnet-base-v2                     | 109M       | 418MB  | 209MB | 105MB | [HF](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 🤗                     |
| multi-qa-mpnet-base-dot-v1            | 109M       | 418MB  | 209MB | 105MB | [HF](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1) 🤗            |
| paraphrase-multilingual-MiniLM-L12-v2 | 118M       | 449MB  | 225MB | 113MB | [HF](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) 🤗 |
| text2vec-base-multilingual            | 118M       | 449MB  | 225MB | 113MB | [HF](https://huggingface.co/shibing624/text2vec-base-multilingual) 🤗                       |
| distiluse-base-multilingual-cased-v1  | 135M       | 514MB  | 257MB | 129MB | [HF](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1) 🤗  |
| paraphrase-multilingual-mpnet-base-v2 | 278M       | 1.04GB | 530MB | 266MB | [HF](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) 🤗 |
| gte-multilingual-base                 | 305M       | 1.17GB | 599MB | 324MB | [HF](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) 🤗                           |
| gte-large                             | 335M       | 1.25GB | 640MB | 321MB | [HF](https://huggingface.co/thenlper/gte-large) 🤗                                          |
| all-roberta-large-v1                  | 355M       | 1.32GB | 678MB | 340MB | [HF](https://huggingface.co/sentence-transformers/all-roberta-large-v1) 🤗                  |
| LaBSE                                 | 470M       | 1.75GB | 898MB | 450MB | [HF](https://huggingface.co/sentence-transformers/LaBSE) 🤗                                 |

**PySentence-Similarity** supports `FP32`, `FP16`, and `INT8` dtypes.

- **FP32:** 32-bit floating-point format that provides high precision and a wide range of values.
- **FP16:** 16-bit floating-point format, reducing memory consumption and computation time, with minimal loss of precision (typically less than 1%).
- **INT8:** 8-bit integer quantized format that greatly reduces model size and speeds up output, ideal for resource-constrained environments, with little loss of precision.

## Usage examples 📖

### Compute similarity score 📊

Let's define the similarity score as the percentage of how similar the sentences are to the original sentence (0.75 = 75%), default compute function is `cosine`

You can use CUDA 12.X by passing the `device='cuda'` parameter to the Model object; the default is `cpu`. If the device is not available, it will automatically be set to `cpu`.

```python
from pysentence_similarity import Model
from pysentence_similarity.utils import compute_score

# Create an instance of the model all-MiniLM-L6-v2; the default dtype is `fp32`
model = Model("all-MiniLM-L6-v2", dtype="fp16")

sentences = [
    "This is another test.",
    "This is yet another test.",
    "We are testing sentence similarity."
]

# Convert sentences to embeddings
# The default is to use mean_pooling as a pooling function
source_embedding = model.encode("This is a test.")
embeddings = model.encode(sentences, progress_bar=True)

# Compute similarity scores
# The rounding parameter allows us to round our float values
# with a default of 2, which means 2 decimal places.
compute_score(source_embedding, embeddings)
# Return: [0.86, 0.77, 0.48]
```

`compute_score` returns in the same index order in which the embedding was encoded.

Let's see the sentence and its evaluation from a computational function

```python
# Compute similarity scores
scores = compute_score(source_embedding, embeddings)

for sentence, score in zip(sentences, scores):
    print(f"{sentence} ({score})")

# Output prints:
# This is another test. (0.86)
# This is yet another test. (0.77)
# We are testing sentence similarity. (0.48)
```

You can use the computational functions: `cosine`, `euclidean`, `manhattan`, `jaccard`, `pearson`, `minkowski`, `hamming`, `kl_divergence`, `chebyshev`, `bregman` or your custom function

```python
from pysentence_similarity.compute import euclidean

compute_score(source_embedding, embeddings, compute_function=euclidean)
# Return: [2.52, 3.28, 5.62]
```

You can use `max_pooling`, `mean_pooling`, `min_pooling` or your custom function

```python
from pysentence_similarity.pooling import max_pooling

source_embedding = model.encode("This is a test.", pooling_function=max_pooling)
embeddings = model.encode(sentences, pooling_function=max_pooling)
...
```

### Search similar sentences 🔍

```python
from pysentence_similarity import Model
from pysentence_similarity.utils import search_similar

# Create an instance of the model
model = Model("all-MiniLM-L6-v2", dtype="fp16")

# Test text
sentences = [
    "Hello my name is Bob.",
    "I love to eat pizza.",
    "We are testing sentence similarity."
    "Today is a sunny day.",
    "London is the capital of England.",
    "I am a student at Stanford University."
]

# Convert query sentence to embedding
query_embedding = model.encode("What's the capital of England?")

# Convert sentences to embeddings
embeddings = model.encode(sentences)

# Search similar sentences
similar = search_similar(
    query_embedding=query_embedding,
    sentences=sentences,
    embeddings=embeddings,
    top_k=3  # number of similar sentences to return
)

# Print similar sentences
for idx, (sentence, score) in enumerate(similar, start=1):
    print(f"{idx}: {sentence} ({score})")

# Output prints:
# 1: London is the capital of England. (0.81)
# 2: Hello my name is Bob. (0.06)
# 3: I love to eat pizza. (0.05)
```

With use storage

```python
from pysentence_similarity import Model, Storage
from pysentence_similarity.utils import search_similar

model = Model("all-MiniLM-L6-v2", dtype="fp16")
query_embedding = model.encode("What's the capital of England?")

storage = Storage.load("my_storage.h5")

similar = search_similar(
    query_embedding=query_embedding,
    storage=storage,
    top_k=3
)
...
```

### Splitting ✂️

```python
from pysentence_similarity import Splitter

# Default split markers: '\n'
splitter = Splitter()

# If you want to separate by specific characters.
splitter = Splitter(markers_to_split=["!", "?", "."], preserve_markers=True)

# Test text
text = "Hello world! How are you? I'm fine."

# Split from text
splitter.split_from_text(text)
# Return: ['Hello world!', 'How are you?', "I'm fine."]
```

At this point, sources for the splitting are available: text, file, URL, CSV, and JSON.

### Storage 💾

The storage allows you to save and link sentences and their embeddings for easy access, so you don't need to encode a large corpus of text every time. The storage also enables similarity searching.

The storage must store the **sentences** themselves and their **embeddings**.

```python
from pysentence_similarity import Model, Storage

# Create an instance of the model
model = Model("all-MiniLM-L6-v2", dtype="fp16")

# Create an instance of the storage
storage = Storage()
sentences = [
    "This is another test.",
    "This is yet another test.",
    "We are testing sentence similarity."
]

# Convert sentences to embeddings
embeddings = model.encode(sentences)

# Add sentences and their embeddings
storage.add(sentences, embeddings)

# Save the storage
storage.save("my_storage.h5")
```

Load from the storage

```python
from pysentence_similarity import Model, Storage
from pysentence_similarity.utils import compute_score

# Create an instance of the model and storage
model = Model("all-MiniLM-L6-v2", dtype="fp16")
storage = Storage.load("my_storage.h5")

# Convert sentence to embedding
source_embedding = model.encode("This is a test.")

# Compute similarity scores with the storage
compute_score(source_embedding, storage)
# Return: [0.86, 0.77, 0.48]
```

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details

<h6 align="center">Created by goldpulpy with ❤️</h6>
