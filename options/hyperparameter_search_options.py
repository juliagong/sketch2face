from .train_options import TrainOptions


class HyperparameterSearchOptions(TrainOptions):
    """This class includes hyperparameter search options.

    It also includes shared options defined in TrainOptions.
    """

    def initialize(self, parser):
        parser = TrainOptions.initialize(self, parser)
        parser.add_argument('--lrs', type=float, nargs='+', default=[0.0001, 0.0002, 0.0004], help='Adam initial learning rates')
        parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4, 16], help='Minibatch sizes')
        parser.add_argument('--beta1s', type=float, nargs='+', default=[0.5, 0.75], help='Adam beta 1 values')
        self.isTrain = True
        return parser
