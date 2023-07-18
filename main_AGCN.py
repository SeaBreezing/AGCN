import argparse
import utils
import torch
import time
from torch.utils.data import DataLoader
import model
import dataloader
import environment
import session


# -------------------------- Hyper Parameter ----------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="AGCN")

    # ----------------------- File Identification
    parser.add_argument('--suffix', type=str, default='amazon_vg', help='生成的日志文件，可视化文件等的文件名后缀')

    # ----------------------- Device Setting
    parser.add_argument('--use_gpu', type=int, default=1, help='是否使用gpu')
    parser.add_argument('--device_id', type=int, default=1, help='若使用gpu，指定gpu编号')
    parser.add_argument('--seed', type=int, default=2021, help='随机数种子')
    parser.add_argument('--process_num', type=int, default=4, help='评估时使用多线程加速，这里指定线程数')

    # ------------------------ Training Setting
    parser.add_argument('--dimension', type=int, default=32, help='free embedding的维数')
    parser.add_argument('--attr_dimension', type=int, default=32, help='属性经过trans后的维数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--update', type=int, default=10, help='更新的最大轮次')
    parser.add_argument('--epoch', type=int, default=500, help='每轮更新的最大迭代次数v')
    parser.add_argument('--batch_size', type=int, default=5120)
    parser.add_argument('--layers', type=int, default=2, help='gcn的层数')
    parser.add_argument('--early_stop', type=int, default=5, help='早停法')
    parser.add_argument('--topk_list', type=str, default='[10]', help='指定评估时的topk')
    parser.add_argument('--neg_num', type=int, default=1, help='每个正样本采样分负样本数量')


    # ----------------------- Regularizer
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--lambda1', type=float, default=0.001)
    parser.add_argument('--lambda2', type=float, default=0.01)

    # ----------------------- Dataset Setting
    parser.add_argument('--dataset', type=str, default='amazon_vg')
    parser.add_argument('--missing_rate', type=float, default=0.9, help='属性缺失的比例')
    parser.add_argument('--dim_list', type=str, default='[14, -10]', help='划分不同的属性')

    # ----------------------- logger
    parser.add_argument('--log', type=int, default=1, help='是否生成日志')
    parser.add_argument('--tensorboard', type=int, default=1, help='是否生成可视化文件')

    return parser.parse_args()


args = parse_args()
utils.cprint(f'{args.suffix}')

# ----------------------------------- Env Init -----------------------------------------------------------

env = environment.Env(args)
utils.cprint('Init Env')

# ----------------------------------- Dataset Init -----------------------------------------------------------

dataset = dataloader.Loader_v1(env)
utils.cprint('Init dataset')

# ----------------------------------- Model&Session Init -----------------------------------------------------------

model = model.AGCN_item(env, dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.process_num)
session = session.Session(env, model, dataloader)

# ---------------------------------------- Main -----------------------------------------------------------

t = time.time()
for update in range(args.update):
    session.train_update(update, args.epoch)
    if update > 0 and session.all_best_ndcg[-1] < session.all_best_ndcg[-2]: # 新的一轮更新没有提升则停止
        utils.cprint(f'jump out the update loop')
        break

# session.train_update(0, args.epoch)
# 打印每轮最优结果
utils.cprint(f'training stage cost time: {time.time() - t}')
utils.cprint(f'---------val hr------------')
for i in session.all_best_hr:
    utils.cprint(f'{i:.5f}')
utils.cprint(f'---------val ndcg------------')
for i in session.all_best_ndcg:
    utils.cprint(f'{i:.5f}')
utils.cprint(f'---------test hr------------')
for i in session.all_test_hr:
    utils.cprint(f'{i:.5f}')
utils.cprint(f'---------test ndcg------------')
for i in session.all_test_ndcg:
    utils.cprint(f'{i:.5f}')

