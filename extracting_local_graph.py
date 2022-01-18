import os
import torch
from torch.utils.data import Dataset
import numpy as np
import csv
import random
import dgl
import itertools


class LocalGraph(Dataset):
    def __init__(self, root, mode, localgraph2label, n_way, k_shot, k_query, batchsz, args, adjs, h):
        self.batchsz = batchsz  # batch of set, not batch of localgraph
        self.n_way = n_way
        self.k_shot = k_shot  # k-shot support set
        self.k_query = k_query  # for query set
        self.setsz = self.n_way * self.k_shot  # num of samples per support set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.h = h  # number of h hops
        self.sample_nodes = args.sample_nodes

        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, %d-hops' % (mode, batchsz, n_way, k_shot, k_query, h))

        self.localgraph2label = localgraph2label

        dictLabels, dictGraphs, dictGraphsLabels = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path

        self.G = []
        for i in adjs:
            self.G.append(i)
        self.localgraph = {}
        self.data_graph = []

        for i, (k, v) in enumerate(dictGraphs.items()):
            self.data_graph.append(v)
        self.graph_num = len(self.data_graph)

        self.data_label = [[] for i in range(self.graph_num)]

        relative_idx_map = dict(
            zip(list(dictGraphs.keys()), range(len(list(dictGraphs.keys())))))

        for i, (k, v) in enumerate(dictGraphsLabels.items()):
            # self.data_label[k] = []
            for m, n in v.items():
                self.data_label[relative_idx_map[k]].append(n)
                # [(graph 1)[(label1)[localgraph1, localgraph2, ...], (label2)[localgraph111, ...]], graph2: [[localgraph1, localgraph2, ...], [localgraph111, ...]] ]

        self.cls_num = len(self.data_label[0])
        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        dictGraphsLabels = {}
        dictLabels = {}
        dictGraphs = {}

        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[1]
                g_idx = int(filename.split('_')[0])
                label = row[2]
                # append filename to current label

                # dictGraphs = {g_idx1 : [filename1, filename2, ...], g_idx : [filename1, filename2, ...], ...}
                if g_idx in dictGraphs.keys():
                    dictGraphs[g_idx].append(filename)
                else:
                    dictGraphs[g_idx] = [filename]
                    dictGraphsLabels[g_idx] = {}

                # dictGraphsLabels = {g_idx1 : {label1 : filename1, ...}, g_idx2 : {label1 : filename1, ...}, ...}
                if label in dictGraphsLabels[g_idx].keys():
                    dictGraphsLabels[g_idx][label].append(filename)
                else:
                    dictGraphsLabels[g_idx][label] = [filename]

                # dictLabels = {label1 : [filename1, filename2, ...], label2 : [filename1, filename2, ...], ...}
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels, dictGraphs, dictGraphsLabels

    def create_batch(self, batchsz):
        """
        create the entire set of batches of tasks for shared label setting, indepedent of # of graphs.
        """
        k_shot = self.k_shot
        k_query = self.k_query

        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # one loop generates one task
            # 1.select n_way classes randomly
            selected_graph = np.random.choice(self.graph_num, 1, False)[0]  # select one graph
            data = self.data_label[selected_graph]

            # for multiple graph setting, we select cls_num * k_shot nodes
            selected_cls = np.array(list(range(len(data))))
            np.random.shuffle(selected_cls)

            support_x = []
            query_x = []

            for cls in selected_cls:

                # 2. select k_shot + k_query for each class
                try:
                    selected_localgraph_idx = np.random.choice(len(data[cls]), k_shot + k_query, False)
                    np.random.shuffle(selected_localgraph_idx)
                    indexDtrain = np.array(selected_localgraph_idx[:k_shot])  # idx for Dtrain
                    indexDtest = np.array(selected_localgraph_idx[k_shot:])  # idx for Dtest
                    support_x.append(
                        np.array(data[cls])[indexDtrain].tolist())  # get all localgraph filename for current Dtrain
                    query_x.append(np.array(data[cls])[indexDtest].tolist())
                except:
                    if len(data[cls]) >= k_shot:
                        selected_localgraph_idx = np.array(range(len(data[cls])))
                        np.random.shuffle(selected_localgraph_idx)
                        indexDtrain = np.array(selected_localgraph_idx[:k_shot])  # idx for Dtrain
                        indexDtest = np.array(selected_localgraph_idx[k_shot:])  # idx for Dtest
                        support_x.append(
                            np.array(data[cls])[indexDtrain].tolist())  # get all localgraph filename for current Dtrain

                        num_more = k_shot + k_query - len(data[cls])
                        count = 0

                        query_tmp = np.array(data[cls])[indexDtest].tolist()

                        while count <= num_more:
                            sub_cls = np.random.choice(selected_cls, 1)[0]
                            idx = np.random.choice(len(data[sub_cls]), 1)[0]
                            query_tmp = query_tmp + [np.array(data[sub_cls])[idx]]
                            count += 1
                        query_x.append(query_tmp)
                    else:
                        print('each class in a graph must have larger than k_shot entities in the current model')

            random.shuffle(support_x)
            random.shuffle(query_x)

            # support_x: [setsz (k_shot+k_query * 1)] numbers of localgraph
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    # helper to generate localgraph on the fly.
    def generate_localgraph(self, G, i, item):
        """
        :param G: self.G[G]
        """
        if item in self.localgraph:
            return self.localgraph[item]
        else:
            # instead of calculating shortest distance, we find the following ways to get localgraph are quicker
            if self.h == 2:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                n_l = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
                h_hops_neighbor = torch.tensor(
                    list(set(list(itertools.chain(*n_l)) + f_hop + [i]))).numpy()
            elif self.h == 1:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                h_hops_neighbor = torch.tensor(list(set(f_hop + [i]))).numpy()
            elif self.h == 3:
                f_hop = [n.item() for n in G.in_edges(i)[0]]
                n_2 = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
                n_3 = [[n.item() for n in G.in_edges(i)[0]] for i in list(itertools.chain(*n_2))]
                h_hops_neighbor = torch.tensor(
                    list(set(list(itertools.chain(*n_2)) + list(itertools.chain(*n_3)) + f_hop + [i]))).numpy()
            if h_hops_neighbor.reshape(-1, ).shape[0] > self.sample_nodes:
                h_hops_neighbor = np.random.choice(h_hops_neighbor, self.sample_nodes, replace=False)
                h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [i]))

            sub = G.subgraph(h_hops_neighbor)
            h_c = list(sub.parent_nid.numpy())
            dict_ = dict(zip(h_c, list(range(len(h_c)))))
            self.localgraph[item] = (
            sub, dict_[i], h_c)

            return sub, dict_[i], h_c

    def __getitem__(self, index):
        """
        get one task. support_x_batch[index], query_x_batch[index]
        index:
        """
        info = [self.generate_localgraph(self.G[int(item.split('_')[0])], int(item.split('_')[1]), item)
                for sublist in self.support_x_batch[index] for item in sublist]

        # obtain a list of DGL localgraph
        support_graph_idx = [int(item.split('_')[0]) for sublist in self.support_x_batch[index] for item in sublist]

        support_x = [i for i, j, k in info]
        support_y = np.array([self.localgraph2label[item]
                              for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        support_center = np.array([j for i, j, k in info]).astype(np.int32)
        support_node_idx = [k for i, j, k in info]

        info = [self.generate_localgraph(self.G[int(item.split('_')[0])], int(item.split('_')[1]), item)
                for sublist in self.query_x_batch[index] for item in sublist]

        # obtain a list of DGL localgraph
        query_graph_idx = [int(item.split('_')[0]) for sublist in self.query_x_batch[index] for item in sublist]

        query_x = [i for i, j, k in info]
        query_y = np.array([self.localgraph2label[item]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        query_center = np.array([j for i, j, k in info]).astype(np.int32)
        query_node_idx = [k for i, j, k in info]

        batched_graph_spt = dgl.batch(support_x)
        batched_graph_qry = dgl.batch(query_x)

        return batched_graph_spt, torch.LongTensor(support_y), batched_graph_qry, torch.LongTensor(
            query_y), torch.LongTensor(support_center), torch.LongTensor(
            query_center), support_node_idx, query_node_idx, support_graph_idx, query_graph_idx

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx = map(list, zip(*samples))

    return graphs_spt, labels_spt, graph_qry, labels_qry, center_spt, center_qry, nodeidx_spt, nodeidx_qry, support_graph_idx, query_graph_idx
