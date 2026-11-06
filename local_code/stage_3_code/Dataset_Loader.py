'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        raw = pickle.load(f)
        f.close()

        X_train, y_train = self._to_array(raw['train'])
        X_test, y_test = self._to_array(raw['test'])

        offset = y_train.min()
        y_train = y_train - offset
        y_test = y_test - offset

        return {'train': {'X': X_train, 'y': y_train},
                'test':  {'X': X_test,  'y': y_test}}

    def _to_array(self, instances):
        X = []
        y = []
        for ins in instances:
            X.append(ins['image'])
            y.append(ins['label'])
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        name = self.dataset_source_file_name
        if name == 'MNIST':
            X = X[:, None, :, :]
        elif name == 'ORL':
            X = X[:, :, :, 0][:, None, :, :]
        elif name == 'CIFAR':
            X = X.transpose(0, 3, 1, 2)
        X = X / 255.0
        return X, y

if __name__ == "__main__":
    for name in ['MNIST', 'ORL', 'CIFAR']:
        loader = Dataset_Loader(name, "")
        loader.dataset_source_folder_path = "data/stage_3_data/"
        loader.dataset_source_file_name = name
        data = loader.load()
        print(name, 'train X:', data['train']['X'].shape, 'test X:', data['test']['X'].shape,
              'label range:', data['train']['y'].min(), '~', data['train']['y'].max())
