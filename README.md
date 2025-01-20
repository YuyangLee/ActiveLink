# Benchmarking Human-in-the-Loop Knowledge Graph Annotation without Human

> Team: Yuntian Gu, Yuyang Li, Rui Yang (Last name A-Z, all equal contribution)

## Prepare environment

```shell
pip install -r requirements.txt
```

## Generate Synthetic Dataset

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
