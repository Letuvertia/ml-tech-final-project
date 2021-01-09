# from __future__ import print_function, division
# from datetime import datetime
# import os
# import numpy as np
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# # from options import ArgumentParser
# # from data_loader import CancelDataset
# # from model.cancel_toy_model import CancelModel
# import yaml

# from evaluate import Grader
# from model import ModelWrapper
# from utils import str_to_bool
# from utils import DataManager, write_test
# from utils import get_revenue_pair, get_label_pair, get_adr_pair

# """


# """

# # Argument parsing
# # args, arg_groups = ArgumentParser().parse()
# # print('Training {} model. Arguments are as follows.'.format(
# #     arg_groups['base']['model']))
# # print(args)
# # print(arg_groups)

# # Seed
# # np.random.seed(arg_groups['base']['seed'])

# np.random.seed(1126)

# import argparse
# parser = argparse.ArgumentParser(description='Hotel Booking Demands Problem')
# parser.add_argument('--tra_path', default='data/train.csv', type=str)
# parser.add_argument('--tst_path', default='data/test.csv', type=str)
# parser.add_argument('--config', default='config/base.yaml', type=str)
# parser.add_argument('--model', type=str, required=True)
# parser.add_argument('--cancel', type=str_to_bool, nargs='?', const=True, default=False, help='predict is_cancelled')
# parser.add_argument('--save_path', default='trained_models', type=str)
# parser.add_argument('--save_name', default='', type=str)
# parser.add_argument('--load_model', type=str, default='', help='load trained model')
# parser.add_argument('--eval', type=str_to_bool, nargs='?', const=True, default=False, help='evaluate the model')
# parser.add_argument('--train', type=str_to_bool, nargs='?', const=True, default=False, help='train the model')


# args = parser.parse_args()

# config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
# print(config)

# DataMgr = DataManager(args.tra_path)
# X_tst = DataMgr.load_test(args.tst_path)
# X_tra = DataMgr.get_feat()
# X_tra, X_val = train_test_split(X_tra, test_size=0.3, random_state=1126)
# grader = Grader(X_val)

# '''
# print('===== training data =====')
# print(type(X_tra), len(X_tra), X_tra[0], X_tra[0][1].shape, X_tra[1][1].shape)
# print('===== testing data =====')
# print(type(X_tst), len(X_tst), X_tst[0], X_tst[0][1].shape, X_tst[1][1].shape)
# '''

# model = ModelWrapper(args.model, config)

# if args.load_model != '':
#     model.load(args.load_model)

# if args.train:
#     print('Starting Training {}'.format(args.model))
#     model.train(X_tra)

#     save_model_name = (args.model + datetime.now().strftime('_%m_%d_%H_%M') if args.save_name == '' else args.save_name) + '.pkl'
#     if not os.path.exists(os.path.join(args.save_path, args.model)):
#         os.makedirs(os.path.join(args.save_path, args.model))
#     save_model_path = os.path.join(args.save_path, args.model, save_model_name)
#     model.save(save_model_path)

# if args.eval:
#     grader = Grader(X_val)
#     #''' validation data
#     print(grader.eval_revenue(model))
#     #print(grader.eval_mae(model))
#     #'''

# #''' testing data
# Y_pre = model.predict(X_tst, output='label')
# write_test(X_tst, Y_pre, args.model+'output.csv')
# #'''


# # if arg_groups['base']['model'] == 'cancel':
# #     # dataset
# #     dataset = CancelDataset(feature_file='train.csv',
# #                             label_file='train_label.csv',
# #                             **arg_groups['dataset'])
# #     data_loader = DataLoader(dataset, batch_size=arg_groups['cancel']['batch_size'])

# #     checkpoint_name = os.path.join(arg_groups['base']['result_model_dir'],
# #                                 arg_groups['base']['result_model_fn'] + '.pth.tar')
# #     print(checkpoint_name)

# #     # model
# #     model = CancelModel(save_model_name=checkpoint_name,
# #                         seed=arg_groups['base']['seed'],
# #                         **arg_groups['cancel'])
# #     model.train_model(train_data_loader=data_loader, val_data_loader=data_loader)
# #     model.test_model(test_data_loader=data_loader)
# #     model.save_model()


from __future__ import print_function, division
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
from utils import DataManager, get_revenue_pair, get_label_pair, get_adr_pair, write_test
from evaluate import Grader
from model import ModelWrapper, CancelModel
import joblib
import yaml

"""


"""


def train(args, config, X_tra, X_val, input_dim=None, onehot_dim=None):
    if args.train_task == 'cancel':
        grader = Grader(X_val)
        model = CancelModel(args.can_model, config,
                            args.filter_all, args.use_onehot)
        if args.train_all:
            X_tra = X_tra + X_val
        
        model.train(X_tra)

        # cacenl error rate a.k.a CER
        cer = grader.eval_cancel_error_rate(model, IsCancelModel=True)
        return model, cer

    elif args.train_task == 'adr' or args.train_task == 'revenue':
        grader = Grader(X_val)
        model = ModelWrapper(
            args, config, args.use_onehot, args.filter_all, input_dim, onehot_dim)

        if args.use_pretrain:
            pretrain_model = ModelWrapper(
                args, config, args.filter_all, args.use_onehot, input_dim, onehot_dim)
            pretrain_model.load('trained_models/pretrain.pkl')
            model.model.model = pretrain_model.model.model

        if args.train_all:
            X_tra = X_tra + X_val
        
        if args.verbose:
            model.train(X_tra, grader)
        else:
            model.train(X_tra)

        # revenue MAE a.k.a REV
        rev = grader.eval_revenue(model)
        mae = grader.eval_mae(model)
        return model, rev, mae
    
    elif args.train_task == 'label':
        grader = Grader(X_val)
        model = ModelWrapper(
            args, config, args.use_onehot, args.filter_all, input_dim, onehot_dim)

        if args.verbose:
            model.train(X_tra, grader)
        else:
            model.train(X_tra)

        rev = grader.eval_revenue(model)
        mae = grader.eval_mae(model)
        return model, rev, mae
    

def pretrain(args, config, X_all, lead_time_idx, input_dim=None, onehot_dim=None):
    model = ModelWrapper(
        args, config, args.use_onehot, input_dim, onehot_dim, lead_time_idx)
    model.train((lead_time_idx, X_all))
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Hotel Booking Demands Problem')
    parser.add_argument('--random_seed', default=1126, type=int)
    parser.add_argument('--tra_path', default='data/pros_train.csv', type=str)
    parser.add_argument('--tst_path', default='data/pros_test.csv', type=str)
    parser.add_argument('--use_onehot', action='store_true')
    parser.add_argument('--filter_all', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--use_pretrain', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--config', default='config/base.yaml', type=str)
    parser.add_argument('--train_task', type=str, required=True)
    parser.add_argument('--can_model', type=str)
    parser.add_argument('--reg_model', type=str)
    parser.add_argument('--can_ckpt', type=str)
    parser.add_argument('--reg_ckpt', type=str)
    parser.add_argument('--val_size', default=0.2, type=float)
    parser.add_argument('--save_path', default='', type=str)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
        
    DataMgr = DataManager(args.tra_path, args.tst_path)
        
    if args.save_path == '':
        args.save_path = args.train_task
    if not os.path.exists(os.path.join('trained_models', args.save_path)):
        os.makedirs(os.path.join('trained_models', args.save_path))

    if args.train_task == 'cancel':
        assert args.can_model is not None
        X_all = DataMgr.get_feat(
            config['base']['cancel_drop_list'], filter_all=args.filter_all, use_onehot=args.use_onehot)
        X_tra, X_val = train_test_split(
            X_all, test_size=args.val_size, random_state=args.random_seed)
        if args.train_all:
            X_tra = X_tra + X_val
        print(X_tra[0][1].shape)
        model, cer = train(args, config, X_tra, X_val)
        model.save(
            'trained_models/{:}/{:}_CER_{:.3f}.pkl'.format(args.save_path, args.can_model, cer))

    else:
        can = CancelModel('RFC', config, args.filter_all, args.use_onehot)
        can.load(args.can_ckpt)
        lst = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15']
        X_all_can = DataMgr.get_feat(
            can.drop_list + lst, filter_all=can.filter_all, use_onehot=can.use_onehot)
        X_all_tar = DataMgr.get_feat(
            config['base']['target_drop_list'], filter_all=args.filter_all, use_onehot=args.use_onehot)
        
        X_tra, X_val_tar, _, X_val_can = train_test_split(
            X_all_tar, X_all_can, test_size=args.val_size, random_state=args.random_seed)

        X_val = (X_val_tar, X_val_can)
 
        if args.train_all:
            X_tra_tar = X_tra_tar + X_val_tar

        # # print(len(X_tra))
        # # X_tra_ = X_tra[:]
        # # for x in X_tra_:
        # #     if (x[0][0] == 2016 and x[0][1] >=6) or (x[0][0] == 2017 and x[0][1] <=3):
        # #         X_tra.append(x)
        # # print(len(X_tra))
        
        model, rev, mae = train(
            args, config, X_tra, X_val, DataMgr.input_dim, DataMgr.onehot_dim)
        model.save(
            'trained_models/{:}/{:}_REV_{:3.3f}_MAE_{:.3f}.pkl'.format(args.save_path, args.reg_model, rev, mae))


if __name__ == "__main__":
    main()
