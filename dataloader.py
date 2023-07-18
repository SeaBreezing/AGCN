import torch
import os
import numpy as np
import scipy.sparse as sp
import random
from collections import defaultdict
import utils


class Loader_v1(torch.utils.data.Dataset):
    """
    数据形式 0位置为userid 之后为user交互的数据
    0 1 2 3 4 5 6 7 ...

    处理单侧的属性
    """

    def __init__(self, env):

        self.env = env

        self.n_user = 0
        self.m_item = 0

        train_file = os.path.join(self.env.DATA_PATH, self.env.args.dataset + '_train.txt')
        val_file = os.path.join(self.env.DATA_PATH, self.env.args.dataset + '_val.txt')
        test_file = os.path.join(self.env.DATA_PATH, self.env.args.dataset + '_test.txt')

        trainUniqueUsers, trainItem, trainUser = [], [], []
        valUniqueUsers, valItem, valUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []

        self.traindataSize = 0
        self.valdataSize = 0
        self.testDataSize = 0

        self.train_data = defaultdict(list)
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.train_data[uid].extend(items)
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))#trainUser存储userid,当前user交互了多少个item就extend多少次当前user
                    trainItem.extend(items)#trainItem存储item矩阵
                    self.m_item = max(self.m_item, max(items))#m_item是某个user交互最多的item个数
                    self.n_user = max(self.n_user, uid)#n_user是user个数
                    self.traindataSize += len(items)#traindataSize是所有user交互过的item总数
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        self.val_data = defaultdict(list)
        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if l[1] == '':
                        continue
                    else:
                        items = [int(i) for i in l[1:]]
                    self.val_data[uid].extend(items)
                    valUniqueUsers.append(uid)
                    valUser.extend([uid] * len(items))
                    valItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.valdataSize += len(items)
        self.valUniqueUsers = np.array(valUniqueUsers)
        self.valUser = np.array(valUser)
        self.valItem = np.array(valItem)

        self.test_data = defaultdict(list)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if l[1] == '':
                        continue
                    else:
                        items = [int(i) for i in l[1:]]
                    self.test_data[uid].extend(items)
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.trainUser_for_sample = self.trainUser.repeat(self.env.args.neg_num)
        self.trainItem_for_sample = self.trainItem.repeat(self.env.args.neg_num)

        attribution_file = os.path.join(self.env.DATA_PATH, self.env.args.dataset + '_item_attribution.txt')
        attrs = []

        with open(attribution_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    attr = [float(i) for i in l]
                    attrs.append(attr)
        self.attr = np.array(attrs)
        self.attr_dim = self.attr.shape[1]

        self.Graph = None


        # self.UserItemNet = self.get_train_adj_matrix(self.train_data)
        self.UserItemNet = sp.csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item))
        #创建稀疏矩阵，有len(self.trainUser)行,每行第一列是user,第二列是item

        # pre-calculate
        self._allPos = self.getUserAllItems(list(range(self.n_user)))

        self.getSparseGraph()
        self._set_missing_attr(self.env.args.missing_rate)

    @property
    def allPos(self):
        return self._allPos

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        将稀疏矩阵转换成稀疏张量
        """
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


    def getSparseGraph(self):
        """
        对邻接矩阵进行归一化，(D−0.5)*A*(D^0.5) 
        构造User-Item图，此时和attribute无关，仅涉及到user-item图，交互了即为1
        """
        if self.Graph is None:

            laplace_file_name = 'laplace_mat.npz'

            try:
                pre_adj_mat = sp.load_npz(os.path.join(self.env.DATA_PATH, laplace_file_name))
                norm_adj = pre_adj_mat


            except:
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
                #dok_matrix采用字典保存矩阵中不为0的元素：字典的键是一个保存元素(行,列)信息的元组，其对应的值为矩阵中位于(行,列)中的元素值
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()#转为字典格式
                rowsum = np.array(adj_mat.sum(axis=1))

                # ---------------------------------agcn
                #  D^{-1} * A开始AGCN了
                d_inv = np.power(rowsum, -1).flatten()
                #flatten返回一个折叠成一维的数组
                #但是该函数只能适用于numpy对象，即array或者mat，普通的list列表是不行的
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                sp.save_npz(os.path.join(self.env.DATA_PATH, laplace_file_name), norm_adj)
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        return self.Graph

    def getUserAllItems(self, users):
        """
        得到训练集和验证集中用户交互过的item
        用于负采样
        """
        posItems = defaultdict(list)
        for user in users:
            # posItems[user].extend(self.test_data[user])
            posItems[user].extend(self.val_data[user])
            #posItems->positive
            posItems[user].extend(self.train_data[user])
        return posItems


    def mask_attr(self, missing_radio=0.9):
        """
        根据missing_radio mask掉一定比例的特征
        不同的属性分开mask
        返回缺失的属性矩阵，属性存在的item的列表，属性缺失的item列表
        """
        all_missing_list, all_existing_list = [], []

        missing_attr = np.zeros(self.attr.shape, dtype=np.float32)
        userset = set(range(self.m_item))

        # 1. random mask users' attributes for evaluate performance
        missing_list_1 = sorted(
            random.sample(list(range(self.m_item)), int(self.m_item * missing_radio)))
        missing_list_2 = sorted(
            random.sample(list(range(self.m_item)), int(self.m_item * missing_radio)))
        missing_list_3 = sorted(
            random.sample(list(range(self.m_item)), int(self.m_item * missing_radio)))
        all_missing_list.append(missing_list_1)
        all_missing_list.append(missing_list_2)
        all_missing_list.append(missing_list_3)

        existing_list_1 = list(userset.symmetric_difference(set(missing_list_1)))
        existing_list_2 = list(userset.symmetric_difference(set(missing_list_2)))
        existing_list_3 = list(userset.symmetric_difference(set(missing_list_3)))
        all_existing_list.append(existing_list_1)
        all_existing_list.append(existing_list_2)
        all_existing_list.append(existing_list_3)

        # 2. padding saved user's attributes with original value
        missing_attr[existing_list_1, :eval(self.env.args.dim_list)[0]] = self.attr[existing_list_1,
                                                                          :eval(self.env.args.dim_list)[0]]
        missing_attr[existing_list_2, eval(self.env.args.dim_list)[0]:eval(self.env.args.dim_list)[1]] = self.attr[
                                                                                                         existing_list_2,
                                                                                                         eval(
                                                                                                             self.env.args.dim_list)[
                                                                                                             0]:eval(
                                                                                                             self.env.args.dim_list)[
                                                                                                             1]]
        missing_attr[existing_list_3, eval(self.env.args.dim_list)[1]:] = self.attr[existing_list_3,
                                                                          eval(self.env.args.dim_list)[1]:]

        # 3. padding masked users' attributes with mean value of saved users'
        mean_attr_1 = np.mean(self.attr[existing_list_1, :eval(self.env.args.dim_list)[0]], axis=0)
        mean_attr_2 = np.mean(
            self.attr[existing_list_2, eval(self.env.args.dim_list)[0]:eval(self.env.args.dim_list)[1]], axis=0)
        mean_attr_3 = np.mean(self.attr[existing_list_3, eval(self.env.args.dim_list)[1]:], axis=0)

        missing_attr[missing_list_1, :eval(self.env.args.dim_list)[0]] = np.tile(mean_attr_1, (len(missing_list_1), 1))
        missing_attr[missing_list_2, eval(self.env.args.dim_list)[0]:eval(self.env.args.dim_list)[1]] = np.tile(
            mean_attr_2, (len(missing_list_2), 1))
        missing_attr[missing_list_3, eval(self.env.args.dim_list)[1]:] = np.tile(mean_attr_3, (len(missing_list_3), 1))

        return missing_attr, all_existing_list, all_missing_list

    def _set_missing_attr(self, missing_radio=0.9):
        self.missing_attr, self.all_existing_list, self.all_missing_list = self.mask_attr(missing_radio)

    def __getitem__(self, index):
        user = self.trainUser_for_sample[index]
        pos_item = self.trainItem_for_sample[index]

        neg_item = np.random.randint(0, self.m_item)

        while neg_item in self.allPos[user]:
            neg_item = np.random.randint(0, self.m_item)

        return user, pos_item, neg_item

    def __len__(self):
        return len(self.trainUser_for_sample)


