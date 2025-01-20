# Benchmarking Human-in-the-Loop Knowledge Graph Annotation without Human

> Team: Yuntian Gu, Yuyang Li, Rui Yang (Last name A-Z, all equal contribution)

## Prepare environment

```shell
pip install -r requirements.txt
```

## Prepare Data

### Prepare Dataset Data

First, go to `data/` and unzip the archives to get the datasets (freebase, wikidata, and synthetic data).
We will only use `Freebase` and synthetic data in this project.

After unzipping, we need to pre-process it:

```shell
python 1_preproc_ds_data.py FB15k-237
python 1_preproc_syn_data.py
```

### Generate Synthetic Dataset (Optional)

We already have generated and pre-processed data under `data/synthetic/`.
If you wish to generate your own synthetic dataset:

```shell
python synthetic_data/generate.py\
 --entity ${ENTITY} \
 --relation ${RELATION} \
 --lambda1 ${LAMBDA1} \
 --lambda2 ${LAMBDA2} \
 --known ${KNOWN} \
 --seed 2024 \
 --output_rule rules.txt \
 --output_relation relations.txt
```

- `entity` and `relation` specify the number of different entities and relations in the knowledge graph. \
- The parameter `lambda1` governs the expectation regarding the number of distinct rules employed to prove a single triplet.
- The parameter `lambda2` governs the expectation of the length of Definite Horn Clause.
- The parameter `known` defines the quantity of initial facts.

After generation, also pre-process the data into the format suitable for our training:

```shell
python 1_preproc_syn_data.py
```

This should generate necessary files under `data/synthetic/`.

## Benchmark KG Annotation (Triplet Selection)

Following ActiveLink, we use **Meta Incremental Learning** to train the KG embeddings incrementally.

```shell
python main.py --dataset DATASET \
               --model MODEL \
               --sampling-mode SAMPLING_MODE \
               --seed SEED \
               --sample-size SAMPLE_SIZE
               --al_epochs EPOCHS
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

The `SAMPLE_SIZE` is the amount of new queries each time.
Use 1000 1024 the real datasets and 128 for synthetic dataset for their differences in sizes.

Also, consider using 100 for the epochs due to the dataset size.
