import importlib


# find the dataset definition by name, for example ScanNetDataset (scannet.py)
def find_dataset_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    if dataset_name in ['scannet', 'scannet_allfrag']:
        return getattr(module, "ScanNetDataset")
    elif dataset_name == 'sevenscenes_allfrag':
        return getattr(module, "Sevenscenes")
