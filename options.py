import argparse

def str_to_bool(v):
    """ This function turn all the True-intended string into True.

    Usage example:
        add_argument('--option1', type=str_to_bool, nargs='?', const=True, default=False, help='...')
    In cmd, all the following lines will be interpreted as option1=True:
        python train.py --option1        # this is set by (nargs='?', const=True)
        python train.py --option1 yes
        python train.py --option1 true
        python train.py --option1 t
        python train.py --option1 y
        python train.py --option1 1
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ArgumentParser():
    def __init__(self, model=None):
        self.parser = argparse.ArgumentParser(description='Hotel Booking Demands Problem')
        self.add_base_parameters()

        # parameters for random forest
        self.add_rf_parameters()

        # parameters for SVM
        self.add_svm_parameters()
        
        # parameters for adaptive boost
        self.add_adaboost_parameters()
        
        # parameters for neural networks
        self.add_nn_parameters()


    # example function of how to add your arguments
    def add_example_parameters(self):
        example_params = self.parser.add_argument_group('example')
        example_params.add_argument('--result-model-fn', type=str, default='trained_model_filename', help='trained model filename')
        example_params.add_argument('--result-model-dir', type=str, default='trained_model', help='path to trained models folder')
        
    

    def add_base_parameters(self):
        base_params = self.parser.add_argument_group('base')
        # add the arguments that will be used by all models
        base_params.add_argument('--model', type=str, default='', help='type of model to use')
        base_params.add_argument('--data-dir', type=str, default='data', help='path to dataset')


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


    def parse(self,arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())
        arg_groups = {}
        for group in self.parser._action_groups:
            if group.title == args.model or group.title == 'base':
                group_dict = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}
                arg_groups[group.title]=group_dict
        return (args,arg_groups)


if __name__ == '__main__':
    """ options.ArgumentParser() usage example

    arg_groups only contains the base arguments and the given model arguments (depend on --model)
    try to run 'python options.py --model svm'
    """
    args,arg_groups = ArgumentParser().parse()
    print('args: ', args)
    print('arg_groups: ', arg_groups)
    print(arg_groups['base']['model'])
