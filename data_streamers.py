# coding=utf-8
from collections import defaultdict
from enum import Enum, Flag
import json
import logging
import os
import random

import numpy as np

from sklearn.cluster import KMeans
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from time import perf_counter

log = logging.getLogger()

class DataStreamer(object):
    def __init__(self, entity2id, rel2id, batch_size, use_all_data=False):
        self.binary_keys = {"e2_multi1"}
        self.multi_entity_keys = {"e2_multi1", "e2_multi2"}
        self.multi_key_length = dict()  #
        self.batch_size = batch_size
        self.data = []  # all triplets
        self.batch_idx = 0
        self.device_id = torch.cuda.current_device()
        self._entity2id = entity2id  # entity to id mapping
        self._rel2id = rel2id  # relation to id mapping
        self.use_all_data = use_all_data  # if True – iterator will return all data (last batch size <= self.batch_size)

        self.str2var = {}
    
    def set_logger(self, logger):
        self.logger = logger
        
    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.rel2id)
    
    @property
    def entity2id(self):
        return self._entity2id
    
    @property
    def rel2id(self):
        return self._rel2id
    
    @property
    def dataset_size(self):
        return len(self.data)
    
    def init_from_path(self, path):
        triplets = []

        with open(path) as f:
            for line in f:
                triple = json.loads(line.strip())
                triple_w_idx = self.tokens_to_ids(triple)
                triplets.append(triple_w_idx)

        self.data = triplets
        self.get_multi_keys_length(triplets)

    def init_from_list(self, triplets):
        self.data = triplets
        self.get_multi_keys_length(triplets)

    def get_multi_keys_length(self, triplets):
        for triple in triplets:
            for key in self.multi_entity_keys:
                if key in triple:
                    assert isinstance(triple[key], list)
                    self.multi_key_length[key] = max(
                        self.multi_key_length.get(key, 0),
                        len(triple[key])
                    )

    def preprocess(self, triplets):
        ent_rel_dict = defaultdict(list)

        # list of dicts -> dict of lists
        for triple in triplets:
            for key, value in triple.items():
                if not isinstance(value, list):
                    new_value = [value]
                else:
                    new_value = value + ([-1] * (self.multi_key_length[key] - len(value)))  # fill in missing values in 2nd dimension
                ent_rel_dict[key].append(new_value)

        # list -> numpy.array
        for key, value in ent_rel_dict.items():
            ent_rel_dict[key] = np.array(value, dtype=np.int64)

        return ent_rel_dict

    def tokens_to_ids(self, triplet):
        entity_keys = {"e1", "e2"}
        multi_entity_keys = {"e2_multi1", "e2_multi2"}
        relation_keys = {"rel", "rel_eval"}

        res = {}

        for key, value in triplet.items():
            if value == "None":
                continue
            if key in entity_keys:
                res[key] = self.entity2id[value]
            elif key in multi_entity_keys:
                res[key] = [self.entity2id[single_value] for single_value in value.split(" ")]
            elif key in relation_keys:
                res[key] = self.rel2id[value]

        return res

    def binary_convertor(self, batch):
        for key in self.binary_keys:
            if key in batch:
                value = batch[key]
                new_value = np.zeros((value.shape[0], self.num_entities), dtype=np.int64)
                for i, row in enumerate(value):
                    for col in row:
                        if col == -1:
                            break
                        new_value[i, col] = 1

                batch[key + "_binary"] = new_value

    def torch_convertor(self, batch):
        for key, value in batch.items():
            batch[key] = Variable(torch.from_numpy(value), volatile=False)

    def torch_cuda_convertor(self, batch):
        for key, value in batch.items():
            batch[key] = value.cuda(self.device_id, True)

    def __iter__(self):
        return self

    def next(self):
        start_index = self.batch_idx * self.batch_size
        if self.use_all_data:
            end_index = min((self.batch_idx + 1) * self.batch_size, self.dataset_size)
        else:
            end_index = (self.batch_idx + 1) * self.batch_size

        if start_index < end_index and end_index <= self.dataset_size:
            self.batch_idx += 1
            current_batch = self.preprocess(self.data[start_index:end_index])
            self.binary_convertor(current_batch)
            self.torch_convertor(current_batch)
            self.torch_cuda_convertor(current_batch)
            return current_batch
        else:
            self.batch_idx = 0
            raise StopIteration
        
    def __next__(self):
        return self.next()

class DataSampleStreamer(DataStreamer):
    def __init__(self, entity_embed_path, entity2id, rel2id, n_clusters, batch_size, sample_size, sampling_mode):
        super(DataSampleStreamer, self).__init__(entity2id, rel2id, batch_size)
        self.entity_embed_path = entity_embed_path
        self.sample_size = sample_size
        self.sampling_mode = sampling_mode
        self.n_clusters = n_clusters
        self.clusters = defaultdict(list)  # {cluster_id: [{"e1": ent1_id, "rel": rel_id, "e2_multi1": [ent2_id, ent3_id]}]}
        self.data = []
        self.remaining_data = []
        

    def init(self, path):
        if self.sampling_mode == "omni_random":
            initial_sample = self.init_random(path)
        elif self.sampling_mode == "omni_t_uncer":
            initial_sample = self.init_random(path)
        elif self.sampling_mode == "random":
            initial_sample = self.init_random(path)
        elif self.sampling_mode == "r_uncer":
            initial_sample = self.init_random(path)
            
        # elif self.sampling_mode == "structured":
        #     initial_sample = self.init_w_clustering(path)
        # elif self.sampling_mode == "structured-uncertainty":
        #     initial_sample = self.init_w_clustering(path)
            
        else:
            raise Exception("Unknown sampling method")

        self.data = initial_sample
        self.update_triplet_map_from_data()
        self.get_multi_keys_length(initial_sample)

        log.info("Training sample size: {}".format(self.dataset_size))

    def init_random(self, path):
        triplets = []
        with open(path) as f:
            for line in f:
                triple = json.loads(line.strip())
                triple_w_ids = self.tokens_to_ids(triple)
                triplets.append(triple_w_ids)

        random.shuffle(triplets)
        sample, self.remaining_data = triplets[:self.sample_size], triplets[self.sample_size:]

        return sample

    def update_triplet_map_from_data(self):
        for triple in self.data:
            e1, rel, e2 = triple["e1"], triple["rel"], triple["e2"]
            self.triplets[e1, rel, e2] = TripletStatus.KNOWN_TRUE
            
            
    def init_w_clustering(self, path):
        self.build_clusters(path)

        empty_clusters = []
        initial_sample = []

        triplets_per_cluster = int(
            round(
                self.sample_size / len(self.clusters)
            )
        )

        if triplets_per_cluster == 0:
            triplets_per_cluster = 1

        stop_sampling = False

        for cluster_id, cluster_data in self.clusters.items():
            if stop_sampling:
                end_index = 0
            else:
                end_index = min(triplets_per_cluster, len(cluster_data))
            random.shuffle(cluster_data)
            initial_sample.extend(cluster_data[:end_index])

            if len(cluster_data) - end_index > 1:  # BatchNorm doesn't accept tensors of length 1
                self.clusters[cluster_id] = cluster_data[end_index:]
            else:
                empty_clusters.append(cluster_id)

            if len(initial_sample) == self.sample_size:
                stop_sampling = True

        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)

        return initial_sample

    @property
    def curr_completeness(self):
        return self.dataset_size / (self.dataset_size + len(self.remaining_data))

    @property
    def curr_n_triplets(self):
        return self.dataset_size

    def update(self, model, logger):
        if self.sampling_mode == "omni_random":
            current_sample = self.update_omni_random()
        elif self.sampling_mode == "omni_t_uncer":
            current_sample = self.update_omni_t_uncert(model)
        elif self.sampling_mode == "random_by_query":
            current_sample = self.update_random()
        elif self.sampling_mode == "relation_uncert_by_query":
            current_sample = self.update_r_uncer()
        # elif self.sampling_mode == "structured":
        #     current_sample = self.update_clustering()
        # elif self.sampling_mode == "structured-uncertainty":
        #     current_sample = self.update_uncert_w_clustering(model)
            
        else:
            raise Exception("Unknown sampling method")

        self.data.extend(current_sample)
        self.get_multi_keys_length(self.data)

        log.info("Training sample size: {}".format(self.dataset_size))

    def update_omni_random(self):
        current_sample, self.remaining_data = self.remaining_data[:self.sample_size], self.remaining_data[self.sample_size:]
        return current_sample

    def update_omni_t_uncert(self, model):
        current_sample = []

        model.train()  # activate dropouts

        if len(self.remaining_data) % self.batch_size == 1:
            batch_size = self.batch_size - 1  # we need this trick because batch_norm doesn't accept tensor of size 1
        else:
            batch_size = self.batch_size

        uncertainty = torch.cuda.FloatTensor(len(self.remaining_data))

        remaining_data_streamer = DataStreamer(self.entity2id, self.rel2id, batch_size, use_all_data=True)
        remaining_data_streamer.init_from_list(self.remaining_data)

        for i, str2var in enumerate(remaining_data_streamer):
            current_batch_size = len(str2var["e1"])

            # init prediciton tensor
            pred = torch.cuda.FloatTensor(10, current_batch_size, self.num_entities)

            for j in range(10):
                pred_ = model.forward(str2var["e1"], str2var["rel"], batch_size=current_batch_size)
                pred[j] = F.sigmoid(pred_).data

            current_batch_uncertainty = self.count_uncertainty(pred)  # 1 x cluster_size
            uncertainty[(i * batch_size): (i * batch_size + current_batch_size)] = current_batch_uncertainty

        uncertainty_sorted, uncertainty_indices_sorted = torch.sort(uncertainty, 0, descending=True)

        top_n = uncertainty_indices_sorted[:self.sample_size]

        for idx in sorted(top_n, reverse=True):  # delete elements from right to left to avoid issues with reindexing
            current_sample.append(self.remaining_data.pop(idx))

        return current_sample
    
    def update_r_uncer(self, model):
        current_sample = []

        model.train()  # activate dropouts

        if len(self.remaining_data) % self.batch_size == 1:
            batch_size = self.batch_size - 1  # we need this trick because batch_norm doesn't accept tensor of size 1
        else:
            batch_size = self.batch_size

        uncertainty = torch.cuda.FloatTensor(len(self.remaining_data))

        remaining_data_streamer = DataStreamer(self.entity2id, self.rel2id, batch_size, use_all_data=True)
        remaining_data_streamer.init_from_list(self.remaining_data)
        
        for i_relation in range(self.num_entities):
            np.where(~self.triplets[:, i_relation])
            
        uncertainty_sorted, uncertainty_indices_sorted = torch.sort(uncertainty, 0, descending=True)

        top_n = uncertainty_indices_sorted[:self.sample_size]

        for idx in sorted(top_n, reverse=True):  # delete elements from right to left to avoid issues with reindexing
            current_sample.append(self.remaining_data.pop(idx))

        return current_sample

    def update_clustering(self):
        empty_clusters = []
        current_sample = []
        all_clusters_size = sum(len(v) for v in self.clusters.values())

        for cluster_id, cluster_data in self.clusters.items():
            random.shuffle(cluster_data)

            current_cluster_ratio = float(len(cluster_data)) / all_clusters_size
            n = int(round(current_cluster_ratio * self.sample_size))

            if n == 0:
                n = 1

            current_sample.extend(cluster_data[:n])

            if len(cluster_data) - n > 1:
                self.clusters[cluster_id] = cluster_data[n:]
            else:
                empty_clusters.append(cluster_id)

        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)

        return current_sample

    def update_uncert_w_clustering(self, model):
        empty_clusters = []
        current_sample = []
        all_clusters_size = sum(len(v) for v in self.clusters.values())

        model.train()  # activate dropouts
        for cluster_id, cluster_data in self.clusters.items():
            if len(cluster_data) % self.batch_size == 1:
                batch_size = self.batch_size - 1  # we need this trick because batch_norm doesn't accept tensor of size 1
            else:
                batch_size = self.batch_size

            uncertainty = torch.cuda.FloatTensor(len(cluster_data))

            cluster_data_streamer = DataStreamer(self.entity2id, self.rel2id, batch_size, use_all_data=True)
            cluster_data_streamer.init_from_list(cluster_data)

            for i, str2var in enumerate(cluster_data_streamer):
                current_batch_size = len(str2var["e1"])

                # init prediciton tensor
                pred = torch.cuda.FloatTensor(10, current_batch_size, self.num_entities)

                for j in range(10):
                    pred_ = model.forward(str2var["e1"], str2var["rel"], batch_size=current_batch_size)
                    pred[j] = F.sigmoid(pred_).data

                current_batch_uncertainty = self.count_uncertainty(pred)  # 1 x cluster_size
                uncertainty[(i * batch_size): (i * batch_size + current_batch_size)] = current_batch_uncertainty

            uncertainty_sorted, uncertainty_indices_sorted = torch.sort(uncertainty, 0, descending=True)

            current_cluster_ratio = float(len(cluster_data)) / all_clusters_size
            n = int(round(current_cluster_ratio * self.sample_size))

            if n == 0:
                n = 1

            top_n = uncertainty_indices_sorted[:n]

            if len(cluster_data) - n <= 1:
                empty_clusters.append(cluster_id)

            for idx in sorted(top_n,
                              reverse=True):  # delete elements from right to left to avoid issues with reindexing
                current_sample.append(cluster_data.pop(idx))

            if len(current_sample) >= self.sample_size:
                break

        for cluster_id in empty_clusters:
            self.clusters.pop(cluster_id)

        return current_sample

    def build_clusters(self, path):
        log.info("Clustering: started")
        entity2cluster = self.do_clusterize()  # {entity_id : cluster_id}

        with open(path) as training_set_file:
            for line in training_set_file:
                triplet = json.loads(line.strip())
                triple_w_ids = self.tokens_to_ids(triplet)
                cluster_id = entity2cluster[triple_w_ids["e1"]]
                self.clusters[cluster_id].append(triple_w_ids)
        log.info("Clustering: finished")

    def do_clusterize(self):
        if not os.path.exists(self.entity_embed_path):
            raise Exception("Entities embedding file is missing")

        labels = {}

        # entity_embeddings = np.loadtxt(self.entity_embed_path)
        weights = torch.load(self.entity_embed_path, weights_only=True)
        entity_embeddings = weights['ent_embeddings.weight'].cpu().numpy()
        relation_embeddings = weights['rel_embeddings.weight'].cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters).fit(entity_embeddings)
        labels_lst = kmeans.labels_.tolist()
        
        # DEBUG - tsne
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        tsne = TSNE(n_components=2, random_state=0)
        entity_embeddings_2d = tsne.fit_transform(entity_embeddings)
        label_colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, self.n_clusters)]
        plt.scatter(entity_embeddings_2d[:, 0], entity_embeddings_2d[:, 1], c=label_colors)
        plt.show()
        

        for entity_id, cluster_id in enumerate(labels_lst):
            labels[entity_id] = cluster_id
        return labels

    def count_uncertainty(self, pred):
        positive = pred
        positive_approx = torch.div(
            torch.sum(positive, 0),
            10
        )

        negative = torch.add(
            torch.neg(positive),
            1
        )
        negative_approx = torch.div(
            torch.sum(negative, 0),
            10
        )

        log_positive_approx = torch.log(positive_approx)
        log_negative_approx = torch.log(negative_approx)

        entropy = torch.neg(
            torch.add(
                torch.mul(positive_approx, log_positive_approx),
                torch.mul(negative_approx, log_negative_approx)
            )
        )

        uncertainty = torch.mean(entropy, 1)
        return uncertainty


class TripletStatus(Flag):
    UNKNOWN_FALSE = 0 # 00
    UNKNOWN_TRUE = 1  # 01
    KNOWN_FALSE = 2   # 10
    KNOWN_TRUE = 3    # 11

class DataTaskStreamer(DataSampleStreamer):
    def __init__(self, entity_embed_path, entity2id, rel2id, n_clusters, batch_size, sample_size, window_size, sampling_mode):
        super(DataTaskStreamer, self).__init__(entity_embed_path, entity2id, rel2id, n_clusters, batch_size, sample_size, sampling_mode) # False is DEBUG
        self.entity_embed_path = entity_embed_path
        self.sample_size = sample_size
        self.window_size = window_size
        self.sampling_mode = sampling_mode
        self.n_clusters = n_clusters
        self.clusters = defaultdict(list)  # {cluster_id: [{"e1": ent1_id, "rel": rel_id, "e2_multi1": [ent2_id, ent3_id]}]}
        self.task_idx = -1
        self.tasks = []
        
    @property
    def dataset_size(self):
        return sum([ task.dataset_size for task in self.tasks ])

    def init_global_triplets(self):
        self.triplets = {}
        # from scipy.sparse import coo_matrix
        # self.triplets = coo_matrix(([], ([], [])), shape=(self.num_entities, self.num_relations, self.num_entities), dtype=np.int8)
        
    # The triplet:
    # Not in the dict -> FALSE
    # In the dict -> TRUE
    def register_triplet_states(self, triplets, is_known=True):
        for triplet in triplets:
            e1, rel = triplet["e1"], triplet["rel"]
            if "e2_multi1" in triplet:
                for e2 in triplet["e2_multi1"]:
                    self.triplets[(e1, rel, e2)] = TripletStatus.KNOWN_TRUE if is_known else TripletStatus.UNKNOWN_TRUE
            else:
                self.triplets[(e1, rel,  triplet["e2"])] = TripletStatus.KNOWN_TRUE if is_known else TripletStatus.UNKNOWN_TRUE
        
    def simulate_query(self, heads, rels, tails, inplace=True):
        added_samples = []
        n_hit = 0
        for head, rel, tail in zip(heads, rels, tails):
            
            if not (head, rel, tail) in self.triplets:
                # n_hit += 1
                # Failed query for being really FALSE
                # self.triplets[(head, rel, tail)] = TripletStatus.KNOWN_FALSE
                continue
            
            if inplace:
                self.triplets[(head, rel, tail)] |= TripletStatus.KNOWN_FALSE # Unkown -> Known
            
            if (self.triplets[(head, rel, tail)] & TripletStatus.UNKNOWN_TRUE).value:
                # n_hit += 1
                added_samples.append({ "e1": head, "rel": rel, "e2_multi1": [tail] })
                # added_samples.append({ "e1": head, "rel": rel, "e2": tail })
            
        return n_hit, added_samples        
    
    def update_random(self):
        current_sample = []
        
        N_SAMPLES_PER_TIME = 10000
        MAX_TURNS = 100000
        
        start_time = perf_counter()
        n_hit = 0
        
        for i in range(MAX_TURNS):
            sampled_heads = np.random.choice(self.num_entities, N_SAMPLES_PER_TIME)
            sampled_rels = np.random.choice(self.num_relations, N_SAMPLES_PER_TIME)
            sampled_tails = np.random.choice(self.num_entities, N_SAMPLES_PER_TIME)
            # valid = (sampled_heads != sampled_tails) * (self.triplets[sampled_heads, sampled_rels, sampled_tails] & StripletStatus.UNKNOWN)        
            valid = (sampled_heads != sampled_tails)
            n_hit_turn, result_queries = self.simulate_query(sampled_heads[valid], sampled_rels[valid], sampled_tails[valid])
            n_hit += n_hit_turn
            
            current_sample += result_queries
            if i % 1000 == 999:
                elapsed_time = perf_counter() - start_time
                curr_pos_hitrate = len(current_sample) / (N_SAMPLES_PER_TIME * (i+1))
                # curr_hitrate = n_hit / (N_SAMPLES_PER_TIME * (i+1))
                print(f"Turn {i + 1}: Totally {len(current_sample)} samples added; PHR = {(curr_pos_hitrate * 1000):.4f}‰; Time elapsed = {elapsed_time:.2f}s")
                # print(f"Turn {i + 1}: Totally {len(current_sample)} samples added; HR = {(curr_pos_hitrate * 1000):.4f}‰ PHR = {(curr_hitrate * 1000):.4f}‰; Time elapsed = {elapsed_time:.2f}s")
            
            if len(current_sample) > self.sample_size:
                elapsed_time = perf_counter() - start_time
                curr_hitrate = len(current_sample) / (N_SAMPLES_PER_TIME * (i+1))
                # curr_pos_hitrate = len(current_sample) / (N_SAMPLES_PER_TIME * (i+1))
                self.logger.log({
                    "annot/time": elapsed_time,
                    "annot/pos_hitrate": curr_pos_hitrate,
                    # "annot/hitrate": curr_hitrate,
                })
                break
            
        return current_sample

    def init(self, path):
        if self.sampling_mode == "omni_random":
            initial_sample = self.init_random(path)
        elif self.sampling_mode == "omni_t_uncer":
            initial_sample = self.init_random(path)
        elif self.sampling_mode == "random":
            initial_sample = self.init_random(path)
            self.init_global_triplets()
            self.register_triplet_states(initial_sample)
            self.register_triplet_states(self.remaining_data, False)
        elif self.sampling_mode == "r_uncer":
            initial_sample = self.init_random(path)
            self.init_global_triplets()
            self.register_triplet_states(initial_sample)
            self.register_triplet_states(self.remaining_data, False)
        elif self.sampling_mode == "structured":
            initial_sample = self.init_w_clustering(path)
        elif self.sampling_mode == "structured-uncertainty":
            initial_sample = self.init_w_clustering(path)
        else:
            raise Exception("Unknown sampling method")
        
        task = DataStreamer(self.entity2id, self.rel2id, self.batch_size)
        task.init_from_list(initial_sample)
        self.tasks.append(task)

        # self.dataset_size = len(initial_sample)

        log.info("Training sample size: {}".format(self.dataset_size))

    def update(self, model):
        if self.sampling_mode == "omni_random":
            current_sample = self.update_omni_random()
        elif self.sampling_mode == "omni_t_uncer":
            current_sample = self.update_omni_t_uncert(model)
        elif self.sampling_mode == "random":
            current_sample = self.update_random()
        elif self.sampling_mode == "relation_uncert":
            current_sample = self.update_r_uncer()
        elif self.sampling_mode == "structured":
            current_sample = self.update_clustering()
        elif self.sampling_mode == "structured-uncertainty":
            current_sample = self.update_uncert_w_clustering(model)
        else:
            raise Exception("Unknown sampling method")

        if len(current_sample) > 0:
            new_task = DataStreamer(self.entity2id, self.rel2id, self.batch_size)
            new_task.init_from_list(current_sample)
            self.tasks.append(new_task)

        # self.dataset_size += len(current_sample)

        log.info("Training sample size: {}".format(self.dataset_size))
        
    def __iter__(self):
        return self
    
    def next(self):
        if self.task_idx * (-1) <= len(self.tasks) and self.task_idx * (-1) <= self.window_size:
            current_task = self.tasks[self.task_idx]
            self.task_idx -= 1
            return current_task
        else:
            self.task_idx = -1
            raise StopIteration
