from local_code.stage_3_code.Method_CNN import Method_CNN

class Method_CNN_ORL(Method_CNN):
    DEFAULTS = dict(conv_channels=(16, 32, 64), kernel_size=3, padding=1, pool='max',
                    hidden_dims=(256,), dropout=0.5, learning_rate=1e-3,
                    max_epoch=40, batch_size=32)

    def __init__(self, mName='CNN-ORL', mDescription='', **overrides):
        super().__init__(mName, mDescription, input_shape=(1, 112, 92), n_classes=40,
                         **{**self.DEFAULTS, **overrides})