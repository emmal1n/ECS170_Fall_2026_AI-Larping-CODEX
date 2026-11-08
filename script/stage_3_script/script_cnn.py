import argparse
import numpy as np, torch
import matplotlib.pyplot as plt
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Setting_CNN import Setting_CNN

METHODS = {'MNIST': Method_CNN_MNIST, 'ORL': Method_CNN_ORL, 'CIFAR': Method_CNN_CIFAR}

def plot_curves(h, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h['epoch'], h['train_loss']); ax[0].set_title('Training loss'); ax[0].set_xlabel('epoch')
    ax[1].plot(h['epoch'], h['train_acc'], label='train')
    ax[1].plot(h['epoch'], h['test_acc'], label='test')
    ax[1].set_title('Accuracy'); ax[1].set_xlabel('epoch'); ax[1].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='MNIST', choices=METHODS)
    p.add_argument('--epochs', type=int, default=None)
    args = p.parse_args()

    np.random.seed(2); torch.manual_seed(2)

    data_obj = Dataset_Loader(args.dataset, '')
    data_obj.dataset_source_folder_path = 'data/stage_3_data/'
    data_obj.dataset_source_file_name = args.dataset

    overrides = {'max_epoch': args.epochs} if args.epochs else {}
    method_obj = METHODS[args.dataset](**overrides)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_3_result/'
    result_obj.result_destination_file_name = f'CNN_{args.dataset}_prediction_result'

    setting_obj = Setting_CNN('train/test', '')
    setting_obj.prepare(data_obj, method_obj, result_obj, Evaluate_Metrics('metrics', ''))
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    for k, v in scores.items():
        print(f'{k}: {v:.4f}')
    plot_curves(result_obj.data['history'], f'CNN on {args.dataset}',
                f'result/stage_3_result/curve_{args.dataset}.png')