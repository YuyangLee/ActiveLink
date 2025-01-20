# Benchmarking Human-in-the-Loop Knowledge Graph Annotation without Human

> Team: Yuntian Gu, Yuyang Li, Rui Yang (Last name A-Z, all equal contribution)

## Prepare environment

```shell
pip install -r requirements.txt
```

## Generate Synthetic Dataset

To generate a complete synthetic dataset, run the following:
```
python synthetic_data/generate.py\
 --entity ${ENTITY}\
 --relation ${RELATION}\
 --lambda1 ${LAMBDA1}\
 --lambda2 ${LAMBDA2}\
 --known ${KNOWN}\
 --seed 2024\
 --output_rule rules.txt\
 --output_relation relations.txt
```

- `entity` and `relation` specify the number of different entities and relations in the knowledge graph. \
- The parameter `lambda1` governs the expectation regarding the number of distinct rules employed to prove a single triplet.
- The parameter `lambda2` governs the expectation of the length of Definite Horn Clause.
- The parameter `known` defines the quantity of initial facts.

## Benchmark KG Annotation (Triplet Selection)

Following ActiveLink, we use **Meta Incremental Learning** to train the KG embeddings incrementally.

```shell
python main.py --dataset DATASET
               --model MODEL
               --sampling-mode SAMPLING_MODE
```

Available `DATASET`: 
- `FB15k-237`: The Freebase15k-237 dataset. Go to `data/` and unzip the three zip archives to get them
- `Synthetic`: Synthetic dataset generated using our algorithm

Available `MODEL`:
- `ConvE`
- `MLP`

Available `SAMPLING_MODE`:
- `omni_random`: random selection with omniscient knowlege (ActiveLink - Random)
- `omni_t_uncer`: triplet-uncertainty sampling with omniscient knowlege (ActiveLink - Triplet Uncertainty)
- `random`: random selection without omniscient knowledge
- `r_uncer`: relation-uncertainty sampling without omniscient knowledge
