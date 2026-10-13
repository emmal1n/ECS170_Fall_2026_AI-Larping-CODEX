'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = "data/stage_2_data"
    dataset_source_file_name = "/train.csv"
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X.append(elements[1:])
            y.append(elements[0])
        f.close()
        return {'X': X, 'y': y}

if __name__ == "__main__":
    loader = Dataset_Loader("test", "")
    loader.dataset_source_folder_path = "data/stage_2_data/"
    loader.dataset_source_file_name = "train.csv"

    data = loader.load()
    print(f"X samples: {len(data['X'])}")
    print(f"y samples: {len(data['y'])}")
