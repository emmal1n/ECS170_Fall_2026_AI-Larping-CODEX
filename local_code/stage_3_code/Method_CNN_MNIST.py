from local_code.stage_3_code.Method_CNN import Method_CNN

class Method_CNN_MNIST(Method_CNN):
    DEFAULTS = dict(conv_channels=(32, 64), kernel_size=3, padding=1, pool='max',
                    hidden_dims=(128,), dropout=0.25, learning_rate=1e-3,
                    max_epoch=10, batch_size=128)

    def __init__(self, mName='CNN-MNIST', mDescription='', **overrides):
        super().__init__(mName, mDescription, input_shape=(1, 28, 28), n_classes=10,
                         **{**self.DEFAULTS, **overrides})

if __name__ == "__main__":
    import torch
    model = Method_CNN_MNIST()
    print(model)                                   
    x = torch.randn(4, 1, 28, 28)               
    print('output shape:', model(x).shape)          
    print('number of parameter:', sum(p.numel() for p in model.parameters()))