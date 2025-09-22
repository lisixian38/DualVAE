import torch
import numpy as np
from cornac.models.recommender import Recommender
from cornac.utils.common import scale
from cornac.exception import ScoreException
from models.dualvae import DualVAE, learn


class DualVAECF(Recommender):
    def __init__(
        self,
        name="DualVAECF",
        k=20,# 潜在空间维度
        a=5,# 方面数量
        encoder_structure=[20], # 编码器隐藏层结构,一层20个神经单元
        decoder_structure=[20],# 解码器网络结构
        act_fn="tanh",# 激活函数类型
        likelihood="pois",# 似然函数类型
        n_epochs=100,# 训练轮数
        batch_size=100,# 批处理大小
        learning_rate=0.001,# 学习率
        beta_kl=1.0,# KL散度损失权重
        gama_cl=0.01,# 对比损失权重
        trainable=True,
        verbose=False,
        seed=None,
        gpu=-1,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)#模型名，是否支持训练，输出详细程度
        self.k = k
        self.a = a
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure
        self.act_fn = act_fn
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta_kl = beta_kl
        self.gama_cl = gama_cl
        self.seed = seed
        self.gpu = gpu

    def fit(self, train_set, val_set=None):#机器学习训练方法名（训练集，验证集）
        Recommender.fit(self, train_set, val_set)
        #cpu or gpu
        self.device = (torch.device("cuda:" + str(self.gpu)) if (self.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu") )

        if self.trainable:

            if self.seed is not None:
                torch.manual_seed(self.seed)# 设置CPU随机种子
                torch.cuda.manual_seed(self.seed)

            if not hasattr(self, "dualvaecf"):
                num_items = train_set.matrix.shape[1]# 物品数量 用户-物品交互矩阵行数
                num_users = train_set.matrix.shape[0]# 用户数量 列数
                self.dualvae = DualVAE(
                    k=self.k,
                    a=self.a,
                    user_encoder_structure=[num_items] + self.encoder_structure,
                    item_encoder_structure=[num_users] + self.encoder_structure,
                    user_decoder_structure=[self.k] + self.decoder_structure,
                    item_decoder_structure=[self.k] + self.decoder_structure,
                    act_fn=self.act_fn,
                    likelihood=self.likelihood,
                ).to(self.device) # 移动到指定设备

            learn(
                self.dualvae, # 要训练的模型
                self.train_set,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learn_rate=self.learning_rate,
                beta_kl=self.beta_kl,
                gama_cl=self.gama_cl,
                verbose=self.verbose,
                device=self.device,
            )

        elif self.verbose:#一个提示
            print("%s is trained already (trainable = False)" % (self.name))

        return self

    def score(self, user_idx, item_idx=None):#（用户索引，物品索引）指定用户计算交互分数

        x = self.train_set.matrix.copy()#二值化有无交互
        x.data = np.ones_like(x.data)  # Binarize data
        tx = x.transpose()#转置矩阵

        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):#检查用户是否在训练集中存在
                raise ScoreException("Can't make score prediction for (user_id=%d)" % user_idx)

            # Reconstructed batch
            known_item_scores = None
            theta = self.dualvae.mu_theta[user_idx]# 用户的方面均值表示
            beta = self.dualvae.mu_beta# 所有物品的方面均值表示
            aspect_prob = self.dualvae.aspect_probability# 物品方面概率
            for a in range(self.a): # 遍历所有方面
                theta_a = theta[a].view(1, -1)# 用户在第a个方面的表示
                beta_a = beta[:, a, :].squeeze()# 所有物品在第a个方面的表示
                aspect_a = aspect_prob[:, a].reshape((1, -1))# 所有物品在第a个方面的概率
                scores_a = self.dualvae.decode_user(theta_a, beta_a) # 解码得到评分
                scores_a = scores_a * aspect_a# 按方面概率加权
                known_item_scores = scores_a if known_item_scores is None else (known_item_scores + scores_a)
            #后处理
            known_item_scores = known_item_scores.detach().cpu().numpy().ravel()
            train_mat = self.train_set.csr_matrix
            csr_row = train_mat.getrow(user_idx)
            pos_items = [item_idx for (item_idx, rating) in zip(csr_row.indices, csr_row.data)]
            known_item_scores[pos_items] = 0. #把训练中已交互物品的分数设为 0，防止已见物品进入推荐候选
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            pred = None
            theta = self.dualvae.mu_theta[user_idx]
            beta = self.dualvae.mu_beta[item_idx]
            aspect_prob = self.dualvae.aspect_probability
            for a in range(self.a):
                theta_a = theta[a, :]# 用户在第a个方面的表示
                beta_a = beta[:, a, :].squeeze() # 物品在第a个方面的表示
                aspect_a = aspect_prob[item_idx, a].reshape((1, -1))# 该物品在第a个方面的概率
                scores_a = self.dualvae.decode_user(theta_a, beta_a)# 解码得到评分
                scores_a = scores_a * aspect_a# 按方面概率加权
                pred = scores_a if pred is None else (pred + scores_a)
            pred = torch.sigmoid(pred).cpu().numpy().ravel()

            pred = scale(pred, self.train_set.min_rating, self.train_set.max_rating, 0.0, 1.0)

            return pred