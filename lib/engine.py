import os
import sys
import time
import copy
import torch
import random
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utils import *
from ranger21 import Ranger
from model.DSTIGFN_new import DSTIGFN


class trainer:
    def __init__(self, scaler, input_dim, num_nodes, channels, dropout, lrate, wdecay, device, granularity, alph, gama, logger):
        self.model = DSTIGFN(
            device, input_dim, num_nodes, channels, granularity, dropout, alph, gama
        )
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = MAE_torch
        self.scaler = scaler
        self.clip = 5
        logger.info("The number of parameters: {}".format(self.model.param_num()))
        logger.info(self.model)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        output = output.transpose(1, 3)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real_val, 0.0)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = MAPE_torch(predict, real_val, 0.0).item()
        rmse = RMSE_torch(predict, real_val, 0.0).item()
        wmape = WMAPE_torch(predict, real_val, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1, 3)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real_val, 0.0)
        mape = MAPE_torch(predict, real_val, 0.0).item()
        rmse = RMSE_torch(predict, real_val, 0.0).item()
        wmape = WMAPE_torch(predict, real_val, 0.0).item()
        return loss.item(), mape, rmse, wmape