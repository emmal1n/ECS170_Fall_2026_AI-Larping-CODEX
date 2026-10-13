from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('stage_2_data', '')
    data_obj.dataset_source_folder_path = 'data/stage_2_data/'
    data_obj.dataset_source_file_name = 'train.csv'

    data_obj_2 = Dataset_Loader('stage_2_data', '')
    data_obj_2.dataset_source_folder_path = 'data/stage_2_data/'
    data_obj_2.dataset_source_file_name = 'test.csv'
    
    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train test', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj,data_obj_2)
    setting_obj.print_setup_summary()
    score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print(f'MLP Accuracy:          {score:.4f}')
    print(f'MLP Precision (macro): {evaluate_obj.PrecisionScore():.4f}')
    print(f'MLP Recall (macro):    {evaluate_obj.RecallScore():.4f}')
    print(f'MLP F1 (macro):        {evaluate_obj.f1Score():.4f}')
    print('************ Finish ************')
    # ------------------------------------------------------
    

    