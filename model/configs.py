_quartznet5x5_config = [
    {'filters': 256, 'repeat': 1, 'kernel': 33, 'stride': 2, 'dilation': 1, 'dropout': 0.2, 'residual': False, 'separable': True},

    {'filters': 256, 'repeat': 5, 'kernel': 33, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},

    {'filters': 256, 'repeat': 5, 'kernel': 39, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': 51, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': 63, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 5, 'kernel': 75, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': True, 'separable': True},

    {'filters': 512, 'repeat': 1, 'kernel': 87, 'stride': 1, 'dilation': 2, 'dropout': 0.2, 'residual': False, 'separable': True},

    {'filters': 1024, 'repeat': 1, 'kernel': 1, 'stride': 1, 'dilation': 1, 'dropout': 0.2, 'residual': False, 'separable': False}
]
