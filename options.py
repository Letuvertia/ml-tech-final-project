import argparse
from utils import str_to_bool


class ArgumentParser():
    def __init__(self, model=None):
        self.parser = argparse.ArgumentParser(description='Hotel Booking Demands Problem')
        self.add_base_parameters() # the arguments that will be used by all models
        self.add_dataset_parameters()
        self.universal_arguments = {'base', 'dataset'}

        # parameters for random forest
        self.add_rf_parameters()

        # parameters for SVM
        self.add_svm_parameters()
        
        # parameters for adaptive boost
        self.add_adaboost_parameters()
        
        # parameters for neural networks
        self.add_nn_parameters()

        # parameters for cancel toy model
        self.add_cancel_model_parameters()
        
    
    def add_base_parameters(self):
        base_params = self.parser.add_argument_group('base')
        base_params.add_argument('--model', type=str, default='', help='type of model to use')
        base_params.add_argument('--use_cuda', type=str_to_bool, nargs='?', const=True, default=False, help='use cuda')
        base_params.add_argument('--result-model-fn', type=str, default='trained_model_filename', help='trained model filename')
        base_params.add_argument('--result-model-dir', type=str, default='trained_models', help='path to trained models folder')
        base_params.add_argument('--seed', type=int, default=1126, help='random seed')


    def add_dataset_parameters(self):
        dataset_params = self.parser.add_argument_group('dataset')
        dataset_params.add_argument('--dataset-csv-path', type=str, default='data', help='path to dataset csv folder')


    def add_rf_parameters(self):
        rf_params = self.parser.add_argument_group('rf')
        # add your rf params
        rf_params.add_argument('--rf_param1', type=float, default=1., help='...')
    

    def add_svm_parameters(self):
        svm_params = self.parser.add_argument_group('svm')
        # add your svm params
        svm_params.add_argument('--svm_param1', type=float, default=1., help='...')
    

    def add_adaboost_parameters(self):
        adaboost_params = self.parser.add_argument_group('adaboost')
        # add your adaboost params
        adaboost_params.add_argument('--ada_param1', type=float, default=1., help='...')
    

    def add_nn_parameters(self):
        nn_params = self.parser.add_argument_group('nn')
        # add your nn params
        nn_params.add_argument('--nn_param1', type=float, default=1., help='...')


    # example function of how to add arguments
    def add_cancel_model_parameters(self):
        cancel_params = self.parser.add_argument_group('cancel')
        cancel_params.add_argument('--input_size', type=int, default=10, help='i.e. feature size (1 dimension)')
        cancel_params.add_argument('--hidden_size', type=int, default=16, help='the size of the hidden layer')
        cancel_params.add_argument('--epoch', type=int, default=1000, help='the number of epoches')
        cancel_params.add_argument('--optimizer', type=str, default='Adam', help='the type of optimizer to use')
        cancel_params.add_argument('--lr', type=float, default=0.001, help='learning rate')
        cancel_params.add_argument('--loss', type=str, default='BCE', help='the type of loss function to use')
        cancel_params.add_argument('--load_trained_model', type=str, default='', help='load_trained_model')
        cancel_params.add_argument('--batch_size', type=int, default=64, help='the size of batch size')


    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())
        arg_groups = {}
        for group in self.parser._action_groups:
            if group.title == args.model or group.title in self.universal_arguments:
                group_dict = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}
                arg_groups[group.title] = group_dict
        return (args, arg_groups)


if __name__ == '__main__':
    """ options.ArgumentParser() usage example

    Note that the dictionary 'arg_groups' only contains the base arguments and \
        the arguments of the given model (depend on --model)
    try to run 'python options.py --model svm'
    """
    args,arg_groups = ArgumentParser().parse()
    print('args: ', args)
    print('arg_groups: ', arg_groups)
    print(arg_groups['base']['model'])
