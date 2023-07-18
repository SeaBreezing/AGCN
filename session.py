import time
import utils
from tqdm import tqdm
import torch
import evaluate
import os


class Session(object):
    def __init__(self, env, model, dataloader):
        self.env = env
        self.model = model
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.env.args.lr)
        self.early_stop = 0 # 用于控制在验证集上的早停

        # 记录每轮中的各种结果
        self.best_ndcg = 0
        self.best_hr = 0
        self.test_ndcg = 0
        self.test_hr = 0
        self.best_epoch = 0
        self.total_epoch = 0
        self.all_best_epoch = []
        self.all_best_ndcg = []
        self.all_best_hr = []
        self.all_test_hr = []
        self.all_test_ndcg = []

    def train_epoch(self):
        """
        一个epoch的训练流程
        """
        t = time.time()
        self.model.train()
        self.total_epoch += 1
        all_loss, all_bpr_loss, all_reg_loss, all_attr_loss = 0., 0., 0., 0.

        for user, pos_item, neg_item in tqdm(self.dataloader):
            user = user.to(self.env.device)
            pos_item = pos_item.to(self.env.device)
            neg_item = neg_item.to(self.env.device)

            bpr_loss, reg_loss_1, reg_loss_2, attr_loss = self.model.loss(user, pos_item, neg_item)
            reg_loss = self.env.args.lambda1 * reg_loss_1 + self.env.args.lambda2 * reg_loss_2

            loss = (bpr_loss + reg_loss) + self.env.args.gamma * attr_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_loss += loss
            all_bpr_loss += bpr_loss
            all_reg_loss += reg_loss
            all_attr_loss += attr_loss

        return all_loss / len(self.dataloader), all_bpr_loss / len(self.dataloader), all_reg_loss / len(
            self.dataloader), all_attr_loss / len(self.dataloader), time.time() - t

    def train_update(self, current_update, epochs):
        """
        整个训练过程由多次update构成，每次update有多个epoch
        """
        self.best_epoch = 0
        self.best_ndcg = 0
        self.best_hr = 0
        self.test_ndcg = 0
        self.test_hr = 0
        self.early_stop = 0

        # 每轮更新开始时根据当前模型为缺失的属性进行预测，重新填充
        if current_update > 0:
            _, items = self.model.compute()

            attribution_missing_1 = self.model.attr_inference(
                items[self.model.dataset.all_missing_list[0]]).detach().cpu().numpy()
            attribution_missing_2 = self.model.attr_inference(
                items[self.model.dataset.all_missing_list[1]]).detach().cpu().numpy()
            attribution_missing_3 = self.model.attr_inference(
                items[self.model.dataset.all_missing_list[2]]).detach().cpu().numpy()

            self.model.dataset.missing_attr[self.model.dataset.all_missing_list[0],
            :eval(self.env.args.dim_list)[0]] = attribution_missing_1[:, :eval(self.env.args.dim_list)[0]]

            self.model.dataset.missing_attr[self.model.dataset.all_missing_list[1],
            eval(self.env.args.dim_list)[0]:eval(self.env.args.dim_list)[1]] = attribution_missing_2[:,
                                                                               eval(self.env.args.dim_list)[0]:
                                                                               eval(self.env.args.dim_list)[1]]

            self.model.dataset.missing_attr[self.model.dataset.all_missing_list[2],
            eval(self.env.args.dim_list)[1]:] = attribution_missing_3[:, eval(self.env.args.dim_list)[1]:]
            utils.cprint(f'update attribution in {current_update}')

        for epoch in range(epochs):
            print(f'EPOCH[{epoch + 1}/{epochs}]')

            # 进行一个epoch
            loss, bpr_loss, reg_loss, attr_loss, train_time = self.train_epoch()
            print(
                f'TRAIN: loss = {loss:.5f}, bpr_loss = {bpr_loss:.5f}, reg_loss = {reg_loss:.5f}, attr_loss = {attr_loss:.5f}, train_time = {train_time:.2f}')

            # 输出验证集的评估结果
            hr, ndcg, val_time = self.val_rec(eval(self.env.args.topk_list))
            for key in hr.keys():
                print(f'hr@{key} = {hr[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

            # 写日志
            if self.env.args.log:
                self.env.train_logger.info(
                    f'EPOCH[{epoch}/{epochs}], loss = {loss:.5f}, bpr_loss = {bpr_loss:.5f}, reg_loss = {reg_loss:.5f}, attr_loss = {attr_loss:.5f}, train_time = {train_time:.2f}')
                self.env.val_logger.info(f'EPOCH[{epoch}/{epochs}]')
                for key in hr.keys():
                    self.env.val_logger.info(
                        f'hr@{key} = {hr[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

            # 生成可视化文件
            if self.env.args.tensorboard:
                self.env.w.add_scalar(f'Train/loss', loss, self.total_epoch)
                self.env.w.add_scalar(f'Train/bpr_loss', bpr_loss, self.total_epoch)
                self.env.w.add_scalar(f'Train/reg_loss', reg_loss, self.total_epoch)
                self.env.w.add_scalar(f'Train/attr_loss', attr_loss, self.total_epoch)

                for key in hr.keys():
                    self.env.w.add_scalar(f'val_rec/hr@{key}', hr[key], self.total_epoch)
                    self.env.w.add_scalar(f'val_rec/ndcg@{key}', ndcg[key], self.total_epoch)

            self.early_stop += 1

            # 更新记录的当前epoch的最好结果
            if ndcg[10] > self.best_ndcg:
                self.early_stop = 0
                self.best_ndcg = ndcg[10]
                self.best_hr = hr[10]
                self.best_epoch = epoch
                if current_update > 0 or epoch > 40:
                    # 测试集上进行评估
                    hr_test, ndcg_test, time_test = self.test_rec()
                    self.test_ndcg = max(self.test_ndcg, ndcg_test[10])
                    self.test_hr = max(self.test_hr, hr_test[10])
                    utils.cprint(
                        f'TEST hr@10 = {hr_test[10]:.5f}, ndcg@10 = {ndcg_test[10]:.5f}, test_time = {time_test:.2f}')

            # 触发早停
            if (current_update == 0 and epoch > 30 and self.early_stop > self.env.args.early_stop) or (
                    current_update > 0 and epoch > 5 and self.early_stop > self.env.args.early_stop):
                # weight_file = f'{self.env.time_stamp}-{current_update}-{self.best_epoch}-{round(self.best_ndcg, 5)}.ckpt'
                # self.save_ckpt(os.path.join(self.env.CKPT_PATH, weight_file))
                # utils.cprint(f'jump out the loop and save model to: {weight_file}')
                self.all_best_epoch.append(self.best_epoch)
                self.all_best_ndcg.append(self.best_ndcg)
                self.all_best_hr.append(self.best_hr)
                self.all_test_hr.append(self.test_hr)
                self.all_test_ndcg.append(self.test_ndcg)
                utils.cprint(f'current_update = {current_update}, best_epoch = {self.best_epoch}')

                break


    def val_rec(self, top_list=[10]):
        """
        用于在验证机集上测试
        """
        self.model.eval()
        t = time.time()
        user_emb = self.model.final_user.cpu().detach().numpy()
        item_emb = self.model.final_item.cpu().detach().numpy()
        hr, ndcg = evaluate.evaluate(self.model.dataset.val_data, self.model.dataset.allPos, top_list,
                                     self.model.dataset.m_item, user_emb, item_emb, self.env.args.process_num)
        return hr, ndcg, time.time() - t

    def val_attr(self):
        """
        用于评估属性推断任务
        """
        self.model.eval()
        t = time.time()
        inferenced_attr = self.model.attr_inference(self.model.final_item).cpu().detach().numpy()

        price_acc = evaluate.get_precise(inferenced_attr[:, :eval(self.env.args.dim_list)[0]],
                                         self.model.dataset.attr[:, :eval(self.env.args.dim_list)[0]])
        platform_map = evaluate.get_map(
            inferenced_attr[:, eval(self.env.args.dim_list)[0]:eval(self.env.args.dim_list)[1]],
            self.model.dataset.attr[:, eval(self.env.args.dim_list)[0]:eval(self.env.args.dim_list)[1]])

        theme_map = evaluate.get_map(inferenced_attr[:, eval(self.env.args.dim_list)[1]:],
                                     self.model.dataset.attr[:, eval(self.env.args.dim_list)[1]:])

        return price_acc, platform_map, theme_map, time.time() - t

    def test_rec(self, top_list=[10]):
        """
        用于在测试机集上测试
        """
        self.model.eval()
        t = time.time()
        user_emb = self.model.final_user.cpu().detach().numpy()
        item_emb = self.model.final_item.cpu().detach().numpy()
        hr, ndcg = evaluate.evaluate(self.model.dataset.test_data, self.model.dataset.allPos, top_list,
                                     self.model.dataset.m_item, user_emb, item_emb, self.env.args.process_num)
        return hr, ndcg, time.time() - t

    def save_ckpt(self, path):
        """
        保存模型
        """
        torch.save(self.model.state_dict(), path)