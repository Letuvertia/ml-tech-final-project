from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from options import ArgumentParser
# from data_loader import CancelDataset
# from model.cancel_toy_model import CancelModel
from utils import DataManeger, get_revenue_pair, get_label_pair, write_test
from evaluate import Grader
from model import ModelWrapper
import yaml

"""


"""

# Argument parsing
# args, arg_groups = ArgumentParser().parse()
# print('Training {} model. Arguments are as follows.'.format(
#     arg_groups['base']['model']))
# print(args)
# print(arg_groups)

# Seed
# np.random.seed(arg_groups['base']['seed'])
np.random.seed(1126)

import argparse
parser = argparse.ArgumentParser(description='Hotel Booking Demands Problem')
parser.add_argument('--tra_path', default='data/train.csv', type=str)
parser.add_argument('--tst_path', default='data/test.csv', type=str)
parser.add_argument('--config', default='config/base.yaml', type=str)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--save_name', default='model/test.pkl', type=str)
args = parser.parse_args()

DataMgr = DataManeger(args.tra_path)
X_tst = DataMgr.load_test(args.tst_path)
X_tra = DataMgr.get_feat()
X_tra, X_val = train_test_split(X_tra, test_size=0.3, random_state=1126)
grader = Grader(X_val)

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
model = ModelWrapper(args.model, config)
model.load('model/RF.pkl')
Y_pre = model.predict(X_tst, output='label')
write_test(X_tst, Y_pre, 'output.csv')
# print(grader.eval_revenue(model))
# print(grader.eval_mae(model))

# if arg_groups['base']['model'] == 'cancel':
#     # dataset
#     dataset = CancelDataset(feature_file='train.csv',
#                             label_file='train_label.csv',
#                             **arg_groups['dataset'])
#     data_loader = DataLoader(dataset, batch_size=arg_groups['cancel']['batch_size'])

#     checkpoint_name = os.path.join(arg_groups['base']['result_model_dir'],
#                                 arg_groups['base']['result_model_fn'] + '.pth.tar')
#     print(checkpoint_name)

#     # model
#     model = CancelModel(save_model_name=checkpoint_name,
#                         seed=arg_groups['base']['seed'],
#                         **arg_groups['cancel'])
#     model.train_model(train_data_loader=data_loader, val_data_loader=data_loader)
#     model.test_model(test_data_loader=data_loader)
#     model.save_model()
