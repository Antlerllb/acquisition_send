# encoding: utf-8
import datasets
import wandb
import os
import pandas as pd
import copy
from typing import Optional
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from util import update_params, init_seed
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    BertModel,
    BertConfig,
    Trainer,
    BertTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding, AdamW, get_scheduler
)


class Config():
    # 基本参与与环境配置
    IS_TRAIN = True
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    SEED = 42
    ENABLE_PROXY = True     # 代理配置
    PROXY = 'http://192.168.207.251:8888'   # 仅当 ENABLE_PROXY 开启时才有效
    ENABLE_WANDB = True     # wandb配置
    WANDB_PROJECT = 'test'  # 仅当 ENABLE_WANDB 开启时才有效
    WANDB_NAME = 'test'     # 仅当 ENABLE_WANDB 开启时才有效

    # 数据
    MAX_LENGTH = 512

    # 模型
    MODEL_REMARK = '简单bert，不冻结'     # 给自己看的备注
    CHECKPOINT = 'google-bert/bert-base-uncased'
    NUM_LABELS = 2
    PROJECTOR_OUTPUT_FEATURES = 128     # 模块中投影头的输出大小
    FROZEN_PARAMS = None    # 需要冻结的参数。会将字符串和 model.names_parameters() 中的 names 进行对比。
    # FROZEN_PARAMS = ['bert']
    
    # 训练
    LR: float = 2e-5
    LR_SCHEDULER_TYPE: str = 'constant'
    TRAIN_BS: int = 4
    EVAL_BS: int = 4
    NUM_TRAIN_EPOCHS = 1
    LOGGING_STEPS = 10  # 记录步数
    WARMUP_STEPS = 0

    # 对比学习
    CON_TEMPERATURE = 0.5
    CON_SCALE_BY_TEMPERATURE = True


class ConLoss(nn.Module):
    """
    https://zhuanlan.zhihu.com/p/442415516
    """
    def __init__(self):
        super(ConLoss, self).__init__()
        self.temperature = Config.CON_TEMPERATURE
        self.scale_by_temperature = Config.CON_SCALE_BY_TEMPERATURE

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1
        输出:
            loss值
        """
        device = Config.DEVICE
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        # 构建mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = (torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) +
                       torch.sum(exp_logits * positives_mask, axis=1, keepdims=True))

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = (torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] /
                     num_positives_per_row[num_positives_per_row > 0])
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


class MyBertModel(nn.Module):
    def __init__(self):
        super(MyBertModel, self).__init__()
        self.num_labels = Config.NUM_LABELS

        # 设置bert的标签数量和隐藏层输出
        bert_config = BertConfig.from_pretrained(Config.CHECKPOINT)
        bert_config.output_hidden_states = True
        bert_config.num_labels = self.num_labels

        # 计算每篇文章可接受的最大长度
        self.per_essay_max_len = int(Config.MAX_LENGTH / 2)

        # 模块
        self.tokenizer = BertTokenizer.from_pretrained(Config.CHECKPOINT)
        self.bert = BertForSequenceClassification.from_pretrained(Config.CHECKPOINT, config=bert_config)
        self.projector = nn.Linear(bert_config.hidden_size, Config.PROJECTOR_OUTPUT_FEATURES)
        self.essay1_reg = nn.Linear(Config.PROJECTOR_OUTPUT_FEATURES * self.per_essay_max_len, 1)
        self.essay2_reg = nn.Linear(Config.PROJECTOR_OUTPUT_FEATURES * self.per_essay_max_len, 1)
        self.con_loss = ConLoss()
        # loss矩阵为 batch_size * num_labels，每个reg矩阵为 batch_size * 1，所以拼接矩阵总输入为 batch_size * 4
        self.fusor = nn.Linear(self.num_labels + 1 + 1, self.num_labels)

    def forward(self, essay1, essay2, label, **kwargs):
        # 不截断。ds_train.csv已经被提前截断过了。
        outputs = self.bert(**self.tokenizer(essay1, essay2, padding=True, return_tensors='pt').to(Config.DEVICE))
        return outputs.logits

    # def forward(self, essay1, essay2, label, **kwargs):
    #     # 基础信息
    #     batch_size = len(essay1)
    #
    #     # Bert 输出
    #     essay1_x = self.bert(**self.tokenizer(essay1, padding='max_length', max_length=self.per_essay_max_len,
    #                                           return_tensors='pt').to(Config.DEVICE)).hidden_states[-1]
    #     essay2_x = self.bert(**self.tokenizer(essay2, padding='max_length', max_length=self.per_essay_max_len,
    #                                           return_tensors='pt').to(Config.DEVICE)).hidden_states[-1]
    #     concat_x = self.bert(**self.tokenizer(essay1, essay2, padding=True,
    #                                           return_tensors='pt').to(Config.DEVICE)).hidden_states[-1]
    #
    #     # 投影头输出
    #     essay1_z = self.projector(essay1_x)
    #     essay2_z = self.projector(essay2_x)
    #     concat_z = self.projector(concat_x)
    #
    #     # 回归两篇文章
    #     essay1_reg_vector = self.essay1_reg(essay1_z.view(batch_size, -1))
    #     essay2_reg_vector = self.essay2_reg(essay2_z.view(batch_size, -1))
    #
    #     # 计算 infonce loss
    #     con_inputs = torch.cat([essay1_z, essay2_z, concat_z], dim=1)
    #     con_inputs = con_inputs.view(con_inputs.size(0), -1)
    #     con_loss = self.con_loss(features=con_inputs, labels=label)
    #     con_loss_matrix = torch.zeros(batch_size, self.num_labels).fill_(con_loss).to(Config.DEVICE)
    #
    #     # 拼接 loss 与 reg 结果，并传入 classifier
    #     fusor_inputs = torch.concat([con_loss_matrix, essay1_reg_vector, essay2_reg_vector], dim=1)
    #     logit = self.fusor(fusor_inputs)
    #     return logit


class ModelCode(object):
    config = Config()

    def __init__(self):
        super().__init__()
        # 初始化种子、代理、wandb
        init_seed(self.config.SEED)
        if Config.ENABLE_PROXY:
            os.environ['HTTP_PROXY'] = Config.PROXY
            os.environ['HTTPS_PROXY'] = Config.PROXY
        if Config.ENABLE_WANDB:
            wandb.init(project=Config.WANDB_PROJECT, name=Config.WANDB_NAME)

        # 初始化模型和数据库的变量
        self.model: Optional[MyBertModel] = None
        self.dl: Optional[DataLoader] = None

        # 初始化输出文件夹
        self.output_dir = Path('./output')
        self.output_model_dir = self.output_dir.joinpath('model')

    def run_model(self):
        self.load_model_and_data()      # 加载模型和数据
        if Config.IS_TRAIN:     # 训练
            self.train()
        else:
            # inference还没写好
            # self.inference()
            ...

    def load_model_and_data(self):
        # 模型加载
        self.model = MyBertModel()
        ds = datasets.load_dataset('csv', data_files='./ds_train.csv', split='train')

        # DataLoader 加载
        batch_size = Config.TRAIN_BS if Config.IS_TRAIN else Config.EVAL_BS
        is_shuffle = True if Config.IS_TRAIN else Config.EVAL_BS
        self.dl = DataLoader(ds, shuffle=is_shuffle, batch_size=batch_size)

    def train(self):
        # 将模型加载到 GPU
        model = self.model
        model.to(Config.DEVICE)

        # 初始化优化器、学习率调度器、损失计算器
        num_training_steps = Config.NUM_TRAIN_EPOCHS * len(self.dl)
        update_params(model, frozen_names=Config.FROZEN_PARAMS)
        optimizer = AdamW(model.parameters(), lr=Config.LR)
        lr_scheduler = get_scheduler(
            Config.LR_SCHEDULER_TYPE,
            optimizer=optimizer,
            num_warmup_steps=Config.WARMUP_STEPS,
            num_training_steps=num_training_steps,
        )
        criterion = nn.CrossEntropyLoss()

        # 开始训练
        model.train()
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(Config.NUM_TRAIN_EPOCHS):
            for step, batch in enumerate(self.dl):
                batch = {k: v for k, v in batch.items()}
                logit = self.model(**batch)
                label = torch.tensor(batch['label'], dtype=torch.long).to(Config.DEVICE)
                loss = criterion(logit, label)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                # 每隔 n 步，报告当前 epoch, lr, loss。
                if step % Config.LOGGING_STEPS == 0:
                    epoch_percent = (epoch * len(self.dl) + step) / num_training_steps
                    lr = lr_scheduler.get_lr()[0]   # 列表中只有一个lr，直接取出来
                    wandb_log = {
                        f'train/epoch': epoch_percent,
                        f'train/lr': lr,
                        f'train/loss': loss.item(),
                    }
                    print(wandb_log)
                    if Config.ENABLE_WANDB:     # 如果开启了wandb，会报告到wandb
                        wandb.log(wandb_log)

        # 保存训练参数
        saved_state_dict = {k: v for k, v in model.state_dict().items()}
        torch.save(saved_state_dict, self.output_model_dir.joinpath('params.pth'))


if __name__ == '__main__':
    code = ModelCode()
    code.run_model()
