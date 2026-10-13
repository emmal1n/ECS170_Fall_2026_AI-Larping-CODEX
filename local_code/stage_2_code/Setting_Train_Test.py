'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import numpy as np

class Setting_Train_Test(setting):
    test_dataset = None
    def prepare(self, sDataset, sMethod, sResult, sEvaluate, sTestDataset=None):
        super().prepare(sDataset, sMethod, sResult, sEvaluate)
        self.test_dataset = sTestDataset

    def load_run_save_evaluate(self):
        
        # load dataset
        train_data = self.dataset.load()
        test_data = self.test_dataset.load()
       
        # run MethodModule
        self.method.data = {'train': {'X': train_data['X'], 'y': train_data['y']}, 'test': {'X': test_data['X'], 'y': test_data['y']}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate()

        