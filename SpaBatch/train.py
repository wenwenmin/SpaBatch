import time

from tqdm import tqdm

import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from torch.autograd import Variable


from adj import *
from model import VGAE_model
from utils import *
from mnn1 import *


class train_model():
    def __init__(self,
                 adata,
                 graph_dict,
                 pre_epochs,
                 epochs,
                 corrupt=0.001,
                 lr=5e-4,
                 weight_decay=1e-4,
                 kl_weight=10,
                 sce_weight=100,
                 bce_kld_weight=0.1,
                 tri_weight=1,
                 mask_rate=0.2,
                 use_gpu=True
                 ):
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.adata = adata
        self.X = adata.obsm['X_pca']
        self.data = torch.FloatTensor(self.X.copy()).to(self.device) #经过数据预处理
        self.input_dim = self.X.shape[1]
        self.adj = graph_dict['adj_norm'].to(self.device) #通过增加自环以及归一化的adj
        self.adj_label = graph_dict['adj_label'].to(self.device) #在adj基础上添加自环
        self.norm = graph_dict['norm_value']
        self.model = VGAE_model(self.input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr = lr, weight_decay = weight_decay)
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.num_spots = self.data.shape[0]
        self.dec_tol = 0
        self.kl_weight = kl_weight
        self.q_stride = 20
        self.tri_stride = 500
        self.sce_weight = sce_weight
        self.bce_kld_weight = bce_kld_weight
        self.tri_weight = tri_weight
        self.corrupt = corrupt
        self.mask_rate = mask_rate


    def train_without_dec(
        self,
        grad_down = 5,
        ):
        #epoch_list = []
        #loss_list = []
        with tqdm(total=int(self.pre_epochs),
                    desc="DeepST_main trains an initial model",
                        bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.pre_epochs):
                # inputs_corr = masking_noise(self.data, self.corrupt) #添加噪声，和data同形的随机噪声
                # inputs_coor = inputs_corr.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                z, mu, logvar, de_feat, _, feat_x, gnn_z, x_init, x_rec = self.model(self.data, self.adj, self.mask_rate)
                preds = self.model.dc(z)
                VGAE_loss = self.model.loss(
                            decoded=de_feat,
                            x=self.data,
                            preds=preds,
                            labels=self.adj_label,
                            mu=mu,
                            logvar=logvar,
                            n_nodes=self.num_spots,
                            norm=self.norm,
                            x_init=x_init,
                            x_rec=x_rec,
                            sce_weight=self.sce_weight,
                            bce_kld_weight=self.bce_kld_weight
                            )
                loss = VGAE_loss
                pbar.set_description(
                    "Epoch {0} total loss={1:.3f}".format(
                        epoch, loss.item()),
                    refresh=True)
                #epoch_list.append(epoch)
                #loss_list.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_down)
                self.optimizer.step()
                pbar.update(1)
        #print(epoch_list)
        #print(loss_list)
        #plot_loss(epoch_list,loss_list,losses_dir=None,datasetname='DLPFC')

    @torch.no_grad()
    def process(
        self,
        ):
        self.model.eval()
        z, _, _, _, q, _, _, _, _ = self.model(self.data, self.adj)
        z = z.cpu().detach().numpy()
        q = q.cpu().detach().numpy()

        return z, q

    def save_model(
        self,
        save_model_file
        ):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(
        self,
        save_model_file
        ):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def train_with_dec(self,
            cluster_n=20,
            clusterType = 'Louvain',
            res = 1.0,
            knn_neigh=100,
            margin=1.0,
            num_aggre=5,
            iter_comb=None,
            ):
        self.train_without_dec()
        pre_z, _ = self.process()
        # z, _, _, _, _, _, _ = self.model(self.data, self.adj)
        # pre_z, _ = self.process()
        if clusterType == 'Kmeans':
            cluster_method = KMeans(n_clusters=cluster_n, n_init=cluster_n * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_method).to(self.device)
        elif clusterType == 'Louvain':
            cluster_data = sc.AnnData(pre_z)
            sc.pp.neighbors(cluster_data, n_neighbors=cluster_n)
            sc.tl.louvain(cluster_data, resolution=res)
            # 将得到的聚类结果保存到cluster_data.obs['louvain']下
            y_pred_last = cluster_data.obs['louvain'].astype(int).to_numpy()
            n_clusters = len(np.unique(y_pred_last))
            features = pd.DataFrame(pre_z, index=np.arange(0, pre_z.shape[0]))
            Group = pd.Series(y_pred_last, index=np.arange(0, features.shape[0]), name="Group")
            Mergefeature = pd.concat([features, Group], axis=1)
            # 分属于每一类的求均值作为簇中心，readout（）函数
            cluster_centers_ = np.asarray(Mergefeature.groupby("Group").mean())

            self.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)

        with tqdm(total=int(self.pre_epochs),
                  desc="DeepST_main trains a final model",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.epochs):
                if epoch % self.q_stride == 0:
                    _, q = self.process()
                    q = self.model.target_distribution(torch.Tensor(q).clone().detach())  # 软分布
                    y_pred = q.cpu().numpy().argmax(1)  # 预测结果，argmax（）找到最大值，当前样本最有可能属于的簇
                    # 计算 delta_label，表示当前预测的聚类标签与上一轮迭代的聚类标签 y_pred_last 的变化程度。
                    # delta_label 是预测结果中标签变化的比例，用于判断聚类结果的稳定性。
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]  #
                    y_pred_last = np.copy(y_pred)
                    self.model.train()
                    if epoch > 0 and delta_label < self.dec_tol:
                        print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                if epoch % self.tri_stride == 0:
                    tri_z, _ = self.process()
                    self.adata.obsm['Tri_SEDR'] = tri_z

                    section_ids = np.array(self.adata.obs['batch_name'].unique())
                    mnn_dict = create_dictionary_mnn_c(self.adata, use_rep='Tri_SEDR', batch_name='batch_name', k=knn_neigh,
                                                     iter_comb=iter_comb, verbose=0)
                    anchor_ind = []
                    positive_ind = []
                    negative_ind = []
                    for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                        batchname_list = self.adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                        #             print("before add KNN pairs, len(mnn_dict[batch_pair]):",
                        #                   sum(adata_new.obs['batch_name'].isin(batchname_list.unique())), len(mnn_dict[batch_pair]))

                        #在每次循环后，cellname_by_batch_dict 字典中的 section_ids[batch_id] 作为键，值是当前批次中所有细胞的名称。
                        cellname_by_batch_dict = dict()
                        for batch_id in range(len(section_ids)):
                            cellname_by_batch_dict[section_ids[batch_id]] = self.adata.obs_names[
                                self.adata.obs['batch_name'] == section_ids[batch_id]].values

                        anchor_list = []
                        positive_list = []
                        negative_list = []
                        for anchor in mnn_dict[batch_pair].keys():
                            anchor_list.append(anchor)
                            ## np.random.choice(mnn_dict[batch_pair][anchor])
                            if num_aggre < 1:
                                raise ValueError("num_aggre must be at least 1.")
                            elif num_aggre == 1:
                                positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                            else:
                                positive_spot = mnn_dict[batch_pair][anchor][:num_aggre]
                            positive_list.append(positive_spot)
                            section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                            negative_list.append(
                                cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])


                        if num_aggre == 1:
                            batch_as_dict = dict(zip(list(self.adata.obs_names), range(0, self.adata.shape[0])))
                            anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                            positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                            negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
                        else:
                            batch_as_dict = dict(zip(list(self.adata.obs_names), range(0, self.adata.shape[0])))
                            for anchor, positives in zip(anchor_list, positive_list):  # anchor_list 和 positive_list 是一一对应的
                                anchor_ind = np.append(anchor_ind, batch_as_dict[anchor])  # 将锚点索引添加到 anchor_ind

                                # 将每个正样本的索引添加到 positive_ind 中
                                # 这里将 positives (一个包含多个正样本的列表) 转化为对应的索引，并将这些正样本索引添加到 positive_ind 中
                                positive_indices = list(map(lambda _: batch_as_dict[_], positives))
                                positive_ind.append(positive_indices)  # 每个锚点的正样本索引作为一行存储

                                # 保持负样本只添加一次
                                negative_ind = np.append(negative_ind,
                                                         batch_as_dict[negative_list[anchor_list.index(anchor)]])  # 负样本的索引

                                anchor_ind = anchor_ind.astype(int)
                                negative_ind = negative_ind.astype(int)

                torch.set_grad_enabled(True)
                self.model.train()
                self.optimizer.zero_grad()
                inputs_coor = self.data.to(self.device)
                z, mu, logvar, de_feat, out_q, feat_x, gnn_z, x_init, x_rec = self.model(self.data, self.adj, self.mask_rate)
                #loss_function = nn.CrossEntropyLoss()
                preds = self.model.dc(z)
                VGAE_loss = self.model.loss(
                    decoded=de_feat,
                    x=self.data,
                    preds=preds,
                    labels=self.adj_label,
                    mu=mu,
                    logvar=logvar,
                    n_nodes=self.num_spots,
                    norm=self.norm,
                    x_init=x_init,
                    x_rec=x_rec,
                    sce_weight=self.sce_weight,
                    bce_kld_weight=self.bce_kld_weight
                )
                kl_loss = F.kl_div(out_q.log(), q.to(self.device))

                if num_aggre == 1:
                    anchor_arr = z[anchor_ind,]
                    positive_arr = z[positive_ind,]
                    negative_arr = z[negative_ind,]

                else:
                    anchor_arr = z[anchor_ind,]
                    negative_arr = z[negative_ind,]
                    positive_arr = read_out(positive_ind, z)
                    positive_arr = torch.tensor(positive_arr, dtype=torch.float32).to(self.device)

                triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
                tri_loss = triplet_loss(anchor_arr, positive_arr, negative_arr)

                loss = self.kl_weight * kl_loss + self.tri_weight * tri_loss + VGAE_loss
                current_loss = loss.cpu().detach().numpy()
                pbar.set_description(
                    "Epoch {0} total loss={1:.3f} recon loss={2:.3f} kl loss={3:.3f} tri loss={4:.3f} ".format(
                        epoch, current_loss, VGAE_loss, kl_loss, tri_loss),
                    refresh=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                pbar.update(1)


def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise