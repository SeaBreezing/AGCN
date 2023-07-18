import os
import time
import torch
import utils
from tensorboardX import SummaryWriter


class Env(object):
    def __init__(self, args):
        """
        初始化环境变量
        设定数据，日志存放位置，时间戳等
        """
        self.args = args

        self.ROOT_PATH = '..'
        self.BASE_PATH = os.path.join(self.ROOT_PATH, 'AGCN')
        self.BASE_PATH = os.path.join(self.BASE_PATH, self.args.dataset)
        self.DATA_PATH = os.path.join(self.ROOT_PATH, 'data')
        self.DATA_PATH = os.path.join(self.DATA_PATH, self.args.dataset)
        self.BOARD_PATH = os.path.join(self.BASE_PATH, 'runs')
        self.CKPT_PATH = os.path.join(self.BASE_PATH, 'checkpoints')
        self.LOG_PATH = os.path.join(self.BASE_PATH, 'log')
        self.PIC_PATH = os.path.join(self.BASE_PATH, 'pic')
        self.time_stamp = time.strftime('%y-%m-%d-%H', time.localtime(time.time()))
        self._check_direcoty()
        self._init_device()

        if self.args.log:
            self._init_logger()

        if self.args.tensorboard:
            self._init_tensorboard()

    def _check_direcoty(self):
        """
        检查环境中的文件夹是否存在，不存在则创建
        """
        if not os.path.exists(self.BASE_PATH):
            os.makedirs(self.BASE_PATH, exist_ok=True)
        if not os.path.exists(self.BOARD_PATH):
            os.makedirs(self.BOARD_PATH, exist_ok=True)
        if not os.path.exists(self.CKPT_PATH):
            os.makedirs(self.CKPT_PATH, exist_ok=True)
        if not os.path.exists(self.LOG_PATH):
            os.makedirs(self.LOG_PATH, exist_ok=True)
        if not os.path.exists(self.PIC_PATH):
            os.makedirs(self.PIC_PATH, exist_ok=True)

    def _init_device(self):
        """
        根据参数和gpu是否可用判断是否使用gpu训练
        """
        if torch.cuda.is_available() and self.args.use_gpu:
            self.device = torch.device(self.args.device_id)
        else:
            self.device = 'cpu'
        utils.cprint(f'Code is running on {self.device}')

    def _init_logger(self):
        """
        创建日志，记录不同阶段的loss等
        """
        utils.cprint(f'Logger Init')
        self.train_logger = utils.get_logger('train', os.path.join(self.LOG_PATH,
                                                              f'{self.time_stamp}_train_log_{self.args.suffix}.log'))
        self.val_logger = utils.get_logger('val',
                                      os.path.join(self.LOG_PATH, f'{self.time_stamp}_val_log_{self.args.suffix}.log'))
        self.test_logger = utils.get_logger('test', os.path.join(self.LOG_PATH,
                                                            f'{self.time_stamp}_test_log_{self.args.suffix}.log'))
        self.train_logger.info(self.args)
        self.val_logger.info(self.args)
        self.test_logger.info(self.args)

    def _init_tensorboard(self):
        """
        初始化tensorboard 用于可视化
        """
        utils.cprint(f'Tensorboard Init')
        self.w = SummaryWriter(os.path.join(self.BOARD_PATH, self.time_stamp + "-" + self.args.suffix))
