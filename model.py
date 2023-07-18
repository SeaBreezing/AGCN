import torch


class AGCN_item(torch.nn.Module):
    """
    属性来自item侧的AGCN模型
    """
    def __init__(self, env, dataset):
        super(AGCN_item, self).__init__()
        self.env = env
        self.dataset = dataset
        self.__init_weight()
        self.to(self.env.device)

    def __init_weight(self):
        self.num_users = self.dataset.n_user
        self.num_items = self.dataset.m_item
        self.latent_dim = self.env.args.dimension # free embedding的维数
        self.attr_dimension = self.env.args.attr_dimension # 特征经过trans后的维数
        self.n_layers = self.env.args.layers # 图神经网络的层数
        self.final_user = None
        self.final_item = None
        self.inferenced_attr = None

        self.graph = self.dataset.Graph.to(self.env.device) # 归一化后的user-item图

        # free embedding， user侧全是free embedding，item侧部分free embedding和trans后的属性拼接而成
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users,
                                                 embedding_dim=self.latent_dim + self.attr_dimension)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items,
                                                 embedding_dim=self.latent_dim)

        # 用于将原始属性转换
        self.trans_layer = torch.nn.Linear(in_features=self.dataset.attr_dim,
                                           out_features=self.attr_dimension, bias=False)

        # 用于属性推断
        self.inference_layer = torch.nn.Linear(in_features=self.latent_dim + self.attr_dimension,
                                               out_features=self.dataset.attr_dim, bias=True)


        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std=0.1)

        torch.nn.init.normal_(self.trans_layer.weight, std=0.01)
        torch.nn.init.normal_(self.inference_layer.weight, std=0.01)

        self.attr_loss = torch.nn.CrossEntropyLoss()

        self.f = torch.nn.Sigmoid()
        self.attr_f = torch.nn.Sigmoid()

    def compute(self):
        """
        前向传播过程
        """
        attr = torch.from_numpy(self.dataset.missing_attr).to(self.env.device)
        attr = self.trans_layer(attr.to(torch.float32))

        users_emb = self.embedding_user.weight
        items_emb = torch.cat(tensors=[self.embedding_item.weight, attr], dim=1)

        emb = torch.cat([users_emb, items_emb])

        for _ in range(self.n_layers):
            new_emb = torch.sparse.mm(self.graph, emb)
            emb = emb + new_emb


        users, items = torch.split(emb, [self.num_users, self.num_items])

        self.final_user = users
        self.final_item = items

        return users, items

    def attr_inference(self, items):
        """
        用于属性推断
        """
        attr = self.inference_layer(items)
        attr = self.attr_f(attr)
        return attr

    def getUsersRating(self, users):
        all_users, all_items = self.compute()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def loss(self, users, pos, neg):
        """
        计算loss
        """

        all_users, all_items = self.compute()
        users_emb = all_users[users]
        pos_emb = all_items[pos]
        neg_emb = all_items[neg]

        # -----------------属性推断loss
        inferenced_attr = self.attr_inference(self.final_item)
        attr_0 = inferenced_attr[self.dataset.all_existing_list[0], :eval(self.env.args.dim_list)[0]]
        attr_1 = inferenced_attr[self.dataset.all_existing_list[1],
                 eval(self.env.args.dim_list)[0]:eval(self.env.args.dim_list)[1]]
        attr_2 = inferenced_attr[self.dataset.all_existing_list[2], eval(self.env.args.dim_list)[1]:]
        attr_loss_0 = torch.mean(self.attr_loss(attr_0, torch.tensor(
            self.dataset.attr[self.dataset.all_existing_list[0], :eval(self.env.args.dim_list)[0]]).to(
            self.env.device).to(torch.float32)))

        attr_loss_1 = torch.mean(self.attr_loss(attr_1, torch.tensor(
            self.dataset.attr[self.dataset.all_existing_list[1],
            eval(self.env.args.dim_list)[0]:eval(self.env.args.dim_list)[1]]).to(self.env.device).to(torch.float32)))

        attr_loss_2 = torch.mean(self.attr_loss(attr_2, torch.tensor(
            self.dataset.attr[self.dataset.all_existing_list[2], eval(self.env.args.dim_list)[1]:]).to(
            self.env.device).to(torch.float32)))

        attr_loss = attr_loss_0 + attr_loss_1 + attr_loss_2



        # userEmb0 = self.embedding_user(users)
        # posEmb0 = self.embedding_item(pos)
        # negEmb0 = self.embedding_item(neg)
        # -----------------正则loss
        reg_loss_1 = (1 / 2) * users_emb.norm(2).pow(2) / float(len(users))

        reg_loss_2 = (1 / 2) * (pos_emb.norm(2).pow(2) +
                                neg_emb.norm(2).pow(2)) / float(len(users))

        # -----------------bpr loss
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return bpr_loss, reg_loss_1, reg_loss_2, attr_loss
