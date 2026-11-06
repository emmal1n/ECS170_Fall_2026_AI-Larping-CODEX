from local_code.base_class.setting import setting

class Setting_CNN(setting):
    def load_run_save_evaluate(self):
        loaded = self.dataset.load()
        self.method.data = loaded
        learned = self.method.run()
        self.result.data = learned
        self.result.save()
        self.evaluate.data = learned
        return self.evaluate.evaluate()