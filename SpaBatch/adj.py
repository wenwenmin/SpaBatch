import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
import networkx as nx
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from torch_sparse import SparseTensor


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca


# #构造feature邻接矩阵
# def construct_graph_by_feature(adata, k=20, mode="connectivity", metric="correlation",
#                                include_self=False):
#     """Constructing feature neighbor graph according to expresss profiles
#         adata.obs['feat']: 低维的基因表达谱
#         k: 邻居数
#         mode: 'connectivity': 构建无权图，邻居之间的连接权重为 1。'distance': 构建加权图，邻居之间的连接权重为欧氏距离或自定义的距离度量。
#         metric: 'euclidean': 欧氏距离。'manhattan': 曼哈顿距离。'cosine': 余弦距离。'minkowski': 闵可夫斯基距离（默认）。
#     """
#
#     feature_graph = kneighbors_graph(adata.obsm['feat'], k, mode=mode, metric=metric,
#                                             include_self=include_self)
#
#     adata.uns['Spatial_feature_graphList'] = feature_graph
#
#     return feature_graph
#
# def Tansfer_feature_Data(adata):
#     adj_feature = torch.FloatTensor(adata.uns['Spatial_feature_graphList'].copy().toarray())
#     #adj_feature = adj_feature + adj_feature.T
#     #adj_feature = np.where(adj_feature > 1, 1, adj_feature)
#     #adj_feature = pre_graph(adj_feature)
#
#     return adj_feature


def construct_graph_by_feature(adata, distType, k_cutoff, verbose=True):

    if verbose:
        print('------Calculating feature graph...')

    adata_high = adata[:, adata.var['highly_variable']]
    adata.obsm['feat'] = pca(adata_high)

    # adj_feat
    if distType == 'Spearmanr':
        from scipy import stats
        SpearA, _ = stats.spearmanr(adata.obsm['feat'], axis=1)
        KNN_list = []
        for node_idx in range(adata.shape[0]):  # 遍历所有节点
            tmp = SpearA[node_idx, :].reshape(1, -1)  # 取每一行的所有元素，重构为一行n列的矩阵（不执行reshape的话只是一个list）
            sorted_idx = tmp.argsort()[0][-(k_cutoff + 1):]
            sorted_idx = sorted_idx[sorted_idx != node_idx]
            res = sorted_idx[:k_cutoff]
            for neighbor in res:
                KNN_list.append((node_idx, neighbor))

    elif distType == 'Cosine':
        from sklearn.metrics import pairwise_distances
        Cos = 1 - pairwise_distances(adata.obsm['feat'], metric='cosine')
        KNN_list = []
        for node_idx in range(adata.shape[0]):  # 遍历所有节点
            tmp = Cos[node_idx, :].reshape(1, -1)  # 取每一行的所有元素，重构为一行n列的矩阵（不执行reshape的话只是一个list）
            sorted_idx = tmp.argsort()[0][-(k_cutoff + 1):]
            sorted_idx = sorted_idx[sorted_idx != node_idx]
            res = sorted_idx[:k_cutoff]
            for neighbor in res:
                KNN_list.append((node_idx, neighbor))

    elif distType == "Kneighbors_graph":
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(adata.obsm['feat'], n_neighbors=k_cutoff, mode='connectivity', metric='correlation', include_self=False)
        A = A.toarray()
        KNN_list = []
        for node_idx in range(adata.shape[0]):
            indices = np.where(A[node_idx] == 1)[0]
            for j in np.arange(0, len(indices)):
                KNN_list.append((node_idx, indices[j]))

    Feature_graphList = pd.DataFrame(KNN_list)
    Feature_Net = Feature_graphList.copy()
    Feature_Net.columns = ['Cell1', 'Cell2']

    id_cell_trans = dict(zip(range(adata.shape[0]), np.array(adata.obs.index), ))
    Feature_Net['Cell1'] = Feature_Net['Cell1'].map(id_cell_trans)
    Feature_Net['Cell2'] = Feature_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' %(Feature_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Feature_Net.shape[0]/adata.n_obs))

    adata.uns['Feature_Net'] = Feature_Net
    adata.uns['Feature_graphList'] = Feature_graphList

    return Feature_graphList

#构造coordinate邻接矩阵，输出为归一化后的adj，原始的adj用于重构adj，以及norm用于计算VGAE中的KL散度
def construct_graph_by_coordinate(adata, distType, k_cutoff, rad_cutoff, verbose=True):

    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    #adj_coordinate
    if distType == 'Radius':
        # 使用sklearn提供的函数计算最近邻,默认值为5，使用coor进行训练,查看结果时使用nbrs.kneighbors()查看参数
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor) #使用rad_cutoff进行计算
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True) #计算出每个点的五个最近邻，以及与他们的距离
        KNN_list = []
        for it in range(indices.shape[0]): #indices.shape[0]=4226
            #KNN_list即构造一个DataFrame，保存从点0,1，...，N，到每个点最近邻以及最近邻的距离
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it])))

    #和上面的方法类似，只不过求最近邻方法不同，该处使用K近邻算法
    if distType == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:])))

    Spatial_graphList = pd.concat(KNN_list)
    Spatial_Net = Spatial_graphList.copy()
    Spatial_Net.columns = ['Cell1', 'Cell2'] #建立列索引，三列索引分别为'Cell1', 'Cell2', 'Distance'，方便后续使用

    #Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,] #保留Distance>0的部分
    # 将range(4226)和barcodes的序列打包成元组，再转换为dict,
    # {0: array(['AAACAACGAATAGTTC-1'], dtype='<U18'),
    #  1: array(['AAACAAGTATCTCCCA-1'], dtype='<U18'),
    #  2: array(['AAACAATCTACTAGCA-1'], dtype='<U18')
    #  ...}
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_graphList.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_graphList.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net
    adata.uns['Spatial_graphList'] = Spatial_graphList

    return Spatial_graphList


def List2Dict(adata, Spatial_graphList):
    """
    Return dict: eg {0: [0, 3542, 2329, 1059, 397, 2121, 485, 3099, 904, 3602],
                 1: [1, 692, 2334, 1617, 1502, 1885, 3106, 586, 3363, 101],
                 2: [2, 1849, 3024, 2280, 580, 1714, 3311, 255, 993, 2629],...}
    """
    #graphList为N*k维矩阵，N为节点数，K为邻居数

    Spatial_graphList_ = [tuple(x) for x in zip(Spatial_graphList[0], Spatial_graphList[1])]
    Spatial_graphdict = {}
    tdict = {}
    for graph in Spatial_graphList_:
        end1 = graph[0] #节点数
        end2 = graph[1] #邻居数（k）
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in Spatial_graphdict:
            tmplist = Spatial_graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        Spatial_graphdict[end1] = tmplist

    for i in range(adata.shape[0]):
        if i not in tdict:
            Spatial_graphdict[i] = []

    return Spatial_graphdict

def mx2SparseTensor(mx):
    """Convert a scipy sparse matrix to a torch SparseTensor.
        将mx(密集矩阵)转换为torch支持的类型
    """
    mx = mx.tocoo().astype(np.float32) #将密集矩阵转化为稀疏矩阵
    row = torch.from_numpy(mx.row).to(torch.long) #取出行
    col = torch.from_numpy(mx.col).to(torch.long) #取出列
    values = torch.from_numpy(mx.data) #取出值
    adj = SparseTensor(row=row, col=col,
                       value=values, sparse_sizes=mx.shape) #转为torch.SparseTensor
    adj_ = adj.t()
    return adj_


'''
def sparse_mx_to_torch_sparse_tensor(mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    shape = torch.Size(mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
'''


def pre_graph(adj):
    """ Graph preprocessing.
        D^(-1/2)*A*D^(-1/2)
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])  # 添加自环,对角线加1
    rowsum = np.array(adj_.sum(1))  # 计算每一行sum
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # 每一行之和先取（-1/2），flatten()变为一行，然后构建一个对角矩阵
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    #return mx2SparseTensor(adj_normalized)  # 将归一化以后的矩阵转成torch.SparseTensor
    return mx2SparseTensor(adj_normalized)  # 将归一化以后的矩阵转成torch.SparseTensor

def main(adata, adj_cons_by, distType, k_cutoff, rad_cutoff):
    if adj_cons_by == 'feature':
        adj_mtx = construct_graph_by_feature(adata, distType=distType, k_cutoff=k_cutoff, verbose=True)
    if adj_cons_by == 'coordinate':
        adj_mtx = construct_graph_by_coordinate(adata, distType=distType, k_cutoff=k_cutoff, rad_cutoff=rad_cutoff, verbose=True)
    graphdict = List2Dict(adata, adj_mtx)  # 转换为Dict
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))  # 转换为 NetworkX 图对象,密集矩阵4226*4226

    """ Store original adjacency matrix (without diagonal entries) for later 
        adj_pre.diagonal()返回对角线元素

    """
    adj_pre = adj_org
    adj_pre = adj_pre - sp.dia_matrix((adj_pre.diagonal()[np.newaxis, :], [0]), shape=adj_pre.shape)  # 去除自环
    adj_pre.eliminate_zeros()  # 稀疏化处理，稀疏矩阵中不会显示值为0的部分

    """ Some preprocessing."""
    adj_norm = pre_graph(adj_pre)  # 归一化后的稀疏邻接矩阵，类型为SparseTensor
    adj_label = adj_pre + sp.eye(adj_pre.shape[0])  # 带有对角线元素的邻接矩阵，用作图自动编码器的标签
    adj_label = torch.FloatTensor(adj_label.toarray())
    # 归一化参数，用于后续的图神经网络操作
    norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)

    Spatial_graph_dict = {
        "adj_norm": adj_norm,
        "adj_label": adj_label,
        "norm_value": norm}

    return Spatial_graph_dict



#STAGATE的邻接矩阵构造方式
from torch_geometric.data import Data

def Transfer_pytorch_Data(adata, key_add):
    #['Spatial_Net','Feature_Net']
    G_df = adata.uns[key_add].copy()
    cells = np.array(adata.obs_names) #通过obs获取barcodes序列，转化为np.array，shape->(4226，)
    cells_id_tran = dict(zip(cells, range(cells.shape[0]))) #打包为一个dic，keys为细胞名，values为其索引
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran) #进行map映射
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    #构建稀疏矩阵,sp.coo_matrix(data,(row,col),shape(N*N)),有连接的置为1，shape->(16904,16904)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0]) #添加自环,对角线全为1

    edgeList = np.nonzero(G) ##可以把G中非0的元素提取出来构造edgeList
    # 转为可训练的Tensor
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data



def combine_graph_dict(dict_1, dict_2):
    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())

    graph_dict = {
        "adj_norm": SparseTensor.from_dense(tmp_adj_norm),
        "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label']),
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])}
    return graph_dict