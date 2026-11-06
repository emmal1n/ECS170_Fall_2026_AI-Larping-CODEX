from local_code.stage_3_code.Method_CNN import Method_CNN

class Method_CNN_CIFAR(Method_CNN):
    DEFAULTS = dict(conv_channels=(64, 128, 256), convs_per_block=2, kernel_size=3,
                    padding=1, pool='max', batch_norm=True, hidden_dims=(256,),
                    dropout=0.5, learning_rate=1e-3, max_epoch=30, batch_size=128,
                    augment=True)

    def __init__(self, mName='CNN-CIFAR', mDescription='', **overrides):
        super().__init__(mName, mDescription, input_shape=(3, 32, 32), n_classes=10,
                         **{**self.DEFAULTS, **overrides})