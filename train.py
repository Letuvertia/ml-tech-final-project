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
# from options import ArgumentParser
import argparse
from utils import DataManager, get_revenue_pair, get_label_pair, write_test, str_to_bool
from evaluate import Grader
from model import ModelWrapper, CancelModel
import yaml


def train(args, config, X_tra, X_val):
    grader = Grader(X_val)

    if args.save_path is None:
        args.save_path = args.train_task

    if args.train_task == 'cancel':
        model = CancelModel(args.can_model, config)
        model.train(X_tra)

        # cacenl error rate a.k.a CER
        cer_acc, cer_0_recall, cer_1_recall = grader.eval_cancel_error_rate(model, IsCancelModel=True)
        return model, cer_acc, cer_0_recall, cer_1_recall

    elif args.train_task == 'adr' or args.train_task == 'revenue':
        model = ModelWrapper(args, config)
        model.train(X_tra)

        # revenue MAE a.k.a REV
        print('Evaluate MAE and REV')
        rev = grader.eval_revenue(model)
        mae = grader.eval_mae(model)
        return model, rev, mae


def main():
    parser = argparse.ArgumentParser(
        description='Hotel Booking Demands Problem')
    parser.add_argument('--random_seed', default=1129, type=int)
    parser.add_argument('--tra_path', default='data/train.csv', type=str)
    parser.add_argument('--tst_path', default='data/test.csv', type=str)
    parser.add_argument('--config', default='config/base.yaml', type=str)
    parser.add_argument('--train_task', type=str, required=True)
    parser.add_argument('--can_model', type=str)
    parser.add_argument('--reg_model', type=str)
    parser.add_argument('--can_ckpt', type=str)
    parser.add_argument('--save_path', type=str, default='', 
        help='name of the directory under /trained_models/')
    parser.add_argument('--grid_search', type=str_to_bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.save_path == '':
        args.save_path = args.train_task
    if not os.path.exists(os.path.join('trained_models', args.save_path)):
        os.makedirs(os.path.join('trained_models', args.save_path))

    print('*Args:')
    print(args, end='\n\n')

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    DataMgr = DataManager(args.tra_path, config['base']['drop_list'], 
        config['base']['poly_trans_list'], config['base']['poly_trans_degree'], config['base']['feature_copy_times'])
    X_all = DataMgr.get_feat()
    #print([x[0] for x in X_all])
    #raise ImportError
    X_tra, X_val = train_test_split(
        X_all, test_size=0.2, random_state=args.random_seed)
    #X_tra, X_val = DataManager.train_test_split_by_date(
    #    X_all, DataMgr.partitions, val_ratio=0.2, random=True)
    print('*dropped column:\n', len(config['base']['drop_list']), config['base']['drop_list'], end='\n\n')

    if args.train_task == 'cancel':
        assert args.can_model is not None

        print('Start training the cancel model {:}'.format(args.can_model))

        if args.grid_search:
            #param_list = {
            #    'n_estimators': [50, 100, 150, 200],
            #    'learning_rate': [0.06, 0.08, 0.1, 0.12]
            #}
            param_list = {
                'max_leaf_nodes': list(range(16, 33, 4)),
                'max_iter': [500, 750]
            }
            param_1, param_2 = (k for k in param_list.keys())
            
            for param_1_val in param_list[param_1]:
                for param_2_val in param_list[param_2]:
                    config_co = config.copy()
                    #print(config_co)
                    config_co[args.can_model][param_1] = param_1_val
                    config_co[args.can_model][param_2] = param_2_val
                    feature_suffix = f'{param_1}_{param_1_val}_{param_2}_{param_2_val}'
                    model, cer_acc, cer_0_recall, cer_1_recall = train(args, config_co, X_tra, X_val)
                    print('cer_acc: ', cer_acc)
                    print('cer_0_recall: ', cer_0_recall)
                    print('cer_1_recall: ', cer_1_recall)
                    model_save_path = 'trained_models/{:}/{:}_CERacc_{:.3f}_CER0rcl_{:.3f}_CER1rcl_{:.3f}_{:}.pkl'.format(args.save_path, args.can_model, cer_acc, cer_0_recall, cer_1_recall, feature_suffix)
                    print('{:} model saved at {:}'.format(args.can_model, model_save_path))
                    model.save(model_save_path)
        else:
            model, cer_acc, cer_0_recall, cer_1_recall = train(args, config, X_tra, X_val)
            print('cer_acc: ', cer_acc)
            print('cer_0_recall: ', cer_0_recall)
            print('cer_1_recall: ', cer_1_recall)
            model_save_path = 'trained_models/{:}/{:}_CERacc_{:.3f}_CER0rcl_{:.3f}_CER1rcl_{:.3f}.pkl'.format(args.save_path, args.can_model, cer_acc, cer_0_recall, cer_1_recall)
            print('{:} model saved at {:}'.format(args.can_model, model_save_path))
            model.save(model_save_path)

    elif args.train_task == 'adr' or args.train_task == 'revenue':
        assert args.reg_model is not None
        
        if args.grid_search:
            param_list = {
                'max_leaf_nodes': [32],
                'max_depth': list(range(10, 31, 2))
            }
            param_1, param_2 = (k for k in param_list.keys())
            
            for param_1_val in param_list[param_1]:
                for param_2_val in param_list[param_2]:
                    config_co = config.copy()
                    config_co[args.reg_model][param_1] = param_1_val
                    config_co[args.reg_model][param_2] = param_2_val
                    feature_suffix = f'{param_1}_{param_1_val}_{param_2}_{param_2_val}'
                    print('Start training the {:} model {:}'.format(args.train_task, args.reg_model))
                    model, rev, mae = train(args, config, X_tra, X_val)
                    model_save_path = 'trained_models/{:}/{:}_MAE_{:.3f}_REV_{:3.3f}_{:}.pkl'.format(args.save_path, args.reg_model, mae, rev, feature_suffix)
                    print('{:} model saved at {:}'.format(args.reg_model, model_save_path))
                    model.save(model_save_path)

        else:
            print('Start training the {:} model {:}'.format(args.train_task, args.reg_model))
            model, rev, mae = train(args, config, X_tra, X_val)

            model_save_path = 'trained_models/{:}/{:}_REV_{:3.3f}_MAE_{:.3f}.pkl'.format(args.save_path, args.reg_model, rev, mae)
            print('{:} model saved at {:}'.format(args.reg_model, model_save_path))
            model.save(model_save_path)


if __name__ == "__main__":
    main()