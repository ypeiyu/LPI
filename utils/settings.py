img_size = 224

parser_choices = {
    'attr_method': ['InputGrad', 'IntGrad', 'ExpGrad', 'AGI', 'LPI', 'Random'],
    'model': ['resnet34', 'vgg16'],
    'dataset': ['ImageNet'],
    'metric': ['DiffID', 'visualize'],
}

parser_default = {
    'attr_method': 'LPI',
    'model': 'resnet34',
    'dataset': 'ImageNet',
    'metric': 'DiffID',
    'k': 5,
    'bg_size': 20,
    'num_centers': 1,
}
