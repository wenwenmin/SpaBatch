import os
import torch
import random
import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
from torch_sparse import SparseTensor
from torch.backends import cudnn
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, contingency_matrix


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=666):
    """\
    该函数用于在Python中使用mclust算法对给定数据进行聚类，并将聚类结果保存在AnnData对象的obs中。
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def Kmeans(adata, num_cluster, use_rep='emb_pca', init='k-means++', n_init=10, max_iter=300, random_state=666):

    data = adata.obsm[use_rep]
    kmeans = KMeans(num_cluster, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state).fit_predict(data)

    adata.obs['kmeans'] = kmeans
    adata.obs['kmeans'] = adata.obs['kmeans'].astype('int')
    adata.obs['kmeans'] = adata.obs['kmeans'].astype('category')

    return adata


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def fx_1NN(i, location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    return np.min(dist_array)

def fx_kNN(i, location_in, k, cluster_in):
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)

    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    else:
        return 0

def _compute_PAS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results) / len(clusterlabel)


def _compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val) / len(clusterlabel)


def compute_CHAOS(adata, pred_key, spatial_key='spatial'):
    return _compute_CHAOS(adata.obs[pred_key], adata.obsm[spatial_key])


def compute_PAS(adata, pred_key, spatial_key='spatial'):
    return _compute_PAS(adata.obs[pred_key], adata.obsm[spatial_key])


def calculate_clustering_matrix(pred, gt, sample, methods_):
    df = pd.DataFrame(columns=['Sample', 'Score', 'Method', "DLPFC"])

    ari = adjusted_rand_score(pred, gt)
    df = df.append(pd.Series([sample, ari, methods_, "Adjusted_Rand_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    nmi = normalized_mutual_info_score(pred, gt)
    df = df.append(pd.Series([sample, nmi, methods_, "Normalized_Mutual_Info_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    hs = homogeneity_score(pred, gt)
    df = df.append(pd.Series([sample, hs, methods_, "Homogeneity_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    purity = purity_score(pred, gt)
    df = df.append(pd.Series([sample, purity, methods_, "Purity_Score"],
                             index=['Sample', 'Score', 'Method', "DLPFC"]), ignore_index=True)

    return df


def refine(
    sample_id,
    pred,
    dis,
    shape="hexagon"
    ):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)

    return refined_pred


def fix_seed(seed):
    # seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


import matplotlib.pyplot as plt
import os


def plot_loss(epochs, losses, losses_dir, datasetname):
    plt.cla()  # 清空当前图形

    # 设置图形的字体和大小
    plt.xlabel("Epoch", fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel("Total Loss", fontdict={'family': 'Times New Roman', 'size': 18})

    # 绘制损失曲线
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b')  # 添加 marker 和颜色

    # 添加标题
    plt.title(f'Loss Curve for {datasetname}', fontdict={'family': 'Times New Roman', 'size': 20})

    # 设置 x 轴和 y 轴的范围（可选）
    plt.xlim(min(epochs), max(epochs))  # x 轴根据 epoch 动态设置范围
    plt.ylim(min(losses) * 0.95, 1000)  # y 轴根据损失值动态设置范围

    plt.grid()  # 添加网格

    # 保存图像到指定目录
    # os.makedirs(losses_dir, exist_ok=True)  # 如果目录不存在，则创建
    # plt.savefig(os.path.join(losses_dir, f'loss_curve_{datasetname}.png'), dpi=300, bbox_inches='tight')  # 保存为 PNG 文件

    plt.show()  # 显示图形



