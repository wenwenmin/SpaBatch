import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import itertools
import networkx as nx
import hnswlib

def read_out(positive_ind, z):
    positive_arr = []

    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()  # 确保 z 是 NumPy 数组

    for i in range(len(positive_ind)):
        selected_vectors_list = []
        for j in positive_ind[i]:
            selected_vectors = z[j, :].reshape(1, -1)  # 维度变换成 1x32
            selected_vectors_list.append(selected_vectors)
        selected_vectors_array = np.vstack(selected_vectors_list)
        mean_vector = np.mean(selected_vectors_array, axis=0).reshape(1, -1)
        positive_arr.append(mean_vector)
    positive_arr = np.array(positive_arr)
    positive_arr = np.squeeze(positive_arr)

    return positive_arr

# Modified from https://github.com/lkmklsmn/insct
def create_dictionary_mnn(adata, use_rep, use_label, batch_name, k = 50, approx = False, verbose = 1, iter_comb = None):
    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    cells = []
    remains_cells = []
    for i in batch_list.unique():
        cells.append(cell_names[batch_list == i])
        remains_cells.append(cell_names[batch_list != i])

    mnns = dict()
    u_unique_set = None
    n_unique_set = None
    for idx, b_name in enumerate(batch_list.unique()):
        key_name = b_name + "_" + "rest"
        mnns[key_name] = {}

        new = list(cells[idx])
        ref = list(remains_cells[idx])

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        gt1 = adata[new].obs[use_label]
        gt2 = adata[ref].obs[use_label]
        names1 = new
        names2 = ref
        match, u_unique_set, n_unique_set = MNN(ds1, ds2, gt1, gt2, names1, names2, u_unique_set, n_unique_set, knn=k)
        if (verbose > 0):
            print('Processing datasets {0} have {1} nodes or edges'.format(b_name, len(match)))

        if len(match) > 0:
            G = nx.Graph()
            G.add_edges_from(match)
            node_names = np.array(G.nodes)
            anchors = list(node_names)
            adj = nx.adjacency_matrix(G)
            tmp = np.split(adj.indices, adj.indptr[1:-1])

            for i in range(0, len(anchors)):
                key = anchors[i]
                i = tmp[i]
                names = list(node_names[i])
                mnns[key_name][key] = names
    return (mnns)


def MNN(target_slice_ds, rest_slice_ds, gt1, gt2, names1, names2, u_unique_set=None, n_unique_set=None, knn=20,
        approx=False):
    if u_unique_set is None:
        u_unique_set = set()
    if n_unique_set is None:
        n_unique_set = set()

    similarity = torch.matmul(torch.tensor(target_slice_ds), torch.transpose(torch.tensor(rest_slice_ds), 1, 0))
    _, I_knn = similarity.topk(k=knn, dim=1, largest=True, sorted=False)

    match_lst = []
    for i in range(I_knn.shape[0]):
        gt = gt1[i]
        gt_tmp = set(gt2[gt2 == gt].index)
        for j in I_knn[i]:
            if names2[j] not in gt_tmp:
                continue
            item = (names1[i], names2[j])
            ex_item = (names2[j], names1[i])

            if ex_item in u_unique_set:
                n_unique_set.add(ex_item)

                continue
            if item not in u_unique_set:
                u_unique_set.add(item)
                match_lst.append(item)

    return match_lst, u_unique_set, n_unique_set

# Modified from https://github.com/lkmklsmn/insct
# def create_dictionary_mnn(adata, use_rep, batch_name, k = 50, save_on_disk = True, approx = True, verbose = 1, iter_comb = None):
#
#     cell_names = adata.obs_names
#
#     batch_list = adata.obs[batch_name]
#     cells = []
#     remains_cells = []
#     for i in batch_list.unique():
#         cells.append(cell_names[batch_list == i])
#         remains_cells.append(cell_names[batch_list != i])
#
#     mnns = dict()
#     u_unique_set = None
#     n_unique_set = None
#     for idx, b_name in enumerate(batch_list.unique()):
#         key_name = b_name + "_" + "rest"
#         mnns[key_name] = {}
#
#         new = list(cells[idx])
#         ref = list(remains_cells[idx])
#
#         ds1 = adata[new].obsm[use_rep]
#         ds2 = adata[ref].obsm[use_rep]
#         names1 = new
#         names2 = ref
#         match, u_unique_set, n_unique_set = MNN(ds1, ds2, names1, names2, u_unique_set, n_unique_set, knn=k)
#         if (verbose > 0):
#             print('Processing datasets {0} have {1} nodes or edges'.format(b_name, len(match)))
#
#         if len(match) > 0:
#             G = nx.Graph()
#             G.add_edges_from(match)
#             node_names = np.array(G.nodes)
#             anchors = list(node_names)
#             adj = nx.adjacency_matrix(G)
#             tmp = np.split(adj.indices, adj.indptr[1:-1])
#
#             for i in range(0, len(anchors)):
#                 key = anchors[i]
#                 i = tmp[i]
#                 names = list(node_names[i])
#                 mnns[key_name][key]= names
#     return(mnns)
#
#
#
#
# def MNN(ds1, ds2, names1, names2, u_unique_set=None, n_unique_set=None, knn = 20, approx = False):
#     if u_unique_set is None:
#         u_unique_set = set()
#     if n_unique_set is None:
#         n_unique_set = set()
#
#     dim = ds2.shape[1]
#     num_elements = ds2.shape[0]
#     p = hnswlib.Index(space='l2', dim=dim)
#     p.init_index(max_elements=num_elements, ef_construction=100, M=16)
#     p.set_ef(10)
#     p.add_items(ds2)
#     ind, distances = p.knn_query(ds1, k=knn)
#
#     match_lst = []
#     for a, b in zip(range(ds1.shape[0]), ind):
#         for b_i in b:
#             item = (names1[a], names2[b_i])
#             ex_item = (names2[b_i], names1[a])
#
#             if ex_item in u_unique_set:
#                 n_unique_set.add(ex_item)
#
#                 continue
#             if item not in u_unique_set:
#                 u_unique_set.add(item)
#                 match_lst.append(item)
#
#     return match_lst, u_unique_set, n_unique_set


# Modified from https://github.com/lkmklsmn/insct
def create_dictionary_mnn_c(adata, use_rep, batch_name, k = 50, save_on_disk = True, approx = True, verbose = 1, iter_comb = None):

    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    mnns = dict()

    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[key_name1] = {} # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair
        if(verbose > 0):
            print('Processing datasets {}'.format((i, j)))

        new = list(cells[j])
        ref = list(cells[i])

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        names1 = new
        names2 = ref
        # if k>1，one point in ds1 may have multiple MNN points in ds2.
        match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk = save_on_disk, approx = approx)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)
        tmp = np.split(adj.indices, adj.indptr[1:-1])

        for i in range(0, len(anchors)):
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            mnns[key_name1][key]= names
    return(mnns)

def validate_sparse_labels(Y):
    if not zero_indexed(Y):
        raise ValueError('Ensure that your labels are zero-indexed')
    if not consecutive_indexed(Y):
        raise ValueError('Ensure that your labels are indexed consecutively')


def zero_indexed(Y):
    if min(abs(Y)) != 0:
        return False
    return True


def consecutive_indexed(Y):
    """ Assumes that Y is zero-indexed. """
    n_classes = len(np.unique(Y[Y != np.array(-1)]))
    if max(Y) >= n_classes:
        return False
    return True


def nn_approx(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M = 16)
    p.set_ef(10)
    p.add_items(ds2)
    ind,  distances = p.knn_query(ds1, k=knn)

    unique_set = set()
    unique_lst = []
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            item = (names1[a], names2[b_i])
            if item not in unique_set:
                unique_set.add(item)
                unique_lst.append(item)
    return unique_set, unique_lst


def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    unique_set = set()
    unique_lst = []
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            item = (names1[a], names2[b_i])
            if item not in unique_set:
                unique_set.add(item)
                unique_lst.append(item)
    return unique_set, unique_lst


def nn_annoy(ds1, ds2, names1, names2, knn = 20, metric='euclidean', n_trees = 50, save_on_disk = True):
    """ Assumes that Y is zero-indexed. """
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    if(save_on_disk):
        a.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def mnn(ds1, ds2, names1, names2, knn = 20, save_on_disk = True, approx = True):
    if approx: 
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1_set, match1_lst = nn_approx(ds1, ds2, names1, names2, knn=knn)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2_set, match2_lst = nn_approx(ds2, ds1, names2, names1, knn=knn)#, save_on_disk = save_on_disk)
    else:
        match1_set, match1_lst  = nn(ds1, ds2, names1, names2, knn=knn)
        match2_set, match2_lst  = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    match2_lst = [(b, a) for a, b in match2_lst]
    # Compute mutual nearest neighbors.
    mutual_set = match1_set & set([(b, a) for a, b in match2_set])
    mutual = [item for item in match2_lst if item in mutual_set]

    return mutual
