# Geometric Deep Learning on Cayley Graphs

Can a graph neural network tell whether a group is abelian just by looking at its Cayley graph?

A Cayley graph Cay(G, S) encodes a group G and a generating set S as a labelled graph. Every vertex has the
same local neighbourhood, which makes these graphs a clean test of what message passing can and cannot learn
about global algebraic structure. This repository has the dataset, code and experiments for that question.

## Results

The task is binary classification (abelian vs non-abelian) on a balanced dataset of **4,412 Cayley graphs**
built with GAP/SageMath.

| Model | Test accuracy |
|---|---|
| Equivariant message passing, constant node features | ~63% (near chance) |
| Message passing with random node features (2 layers) | ~92% |
| Equivariant message passing, anchored at the identity (δ_e initialisation) | ~93.7% |
| **Path-indexed state-space model on words in the generators** | **~95%** |

Why the baseline fails: in a Cayley graph every vertex looks the same, so a symmetric message-passing network
receives identical messages everywhere and cannot separate vertices. Anchoring the signal at the identity
breaks the symmetry canonically while keeping translation equivariance. The path model works on words in the
generators instead of vertices, so relations such as `ab = ba` are visible directly in its input.

On the theory side, translation-equivariant linear layers on Cay(G, S) are exactly left k[G]-module
homomorphisms (group convolutions), and equivariance under the full automorphism group of the labelled graph
restricts the kernels further.

## Repository layout

```
notebooks/
  01_build_dataset_sage.ipynb               build the Cayley graph datasets (SageMath kernel)
  02_first_experiments.ipynb                early exploration with PyTorch Geometric
  03_abelian_classification_baselines.ipynb baseline GNNs and evaluation
  04_equivariant_message_passing.ipynb      equivariant, edge-aware and anchored message passing
  05_path_model.ipynb                       path-indexed (relation-space) model
scripts/
  CayleyGraphSage.py                        Cayley graph of a finite group in Sage
  build_cayley_dataset.sage                 standalone dataset generator (older schema)
data/                                       datasets as gzipped JSON Lines
```

## Getting started

```bash
pip install -r requirements.txt
gunzip -k data/*.jsonl.gz      # notebooks read the uncompressed .jsonl files from data/
cd notebooks && jupyter lab
```

Notebook 01 needs [SageMath](https://www.sagemath.org/) (with GAP). You only need it to regenerate the data.
Notebooks 02 to 05 run in plain Python.

## Data format

Each line of a `.jsonl` file is one labelled Cayley graph:

```json
{"edge_index": [[0, 1], [1, 0]], "edge_label": [1, 1], "num_generators": 1,
 "num_vertices": 2, "num_edges": 2, "is_abelian": 1, "group_id": "C2"}
```

`edge_index` lists directed edges g → g·s, and `edge_label` gives the generator s for each edge. Files named
`unique` contain one generating set per group. Files named `five_gens_each` and `ten_gens_each` sample several
generating sets per group. The suffixes give the range of group orders.

## Author

Utkarsh Bajaj, University of Oxford. A paper based on this project is in preparation.
