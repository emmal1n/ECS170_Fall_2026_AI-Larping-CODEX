import argparse, time, csv
import numpy as np, torch
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import pickle
import os

METHODS = {'MNIST': Method_CNN_MNIST, 'ORL': Method_CNN_ORL, 'CIFAR': Method_CNN_CIFAR}

def get_variants(D):
    return {
        'baseline':          {},
        'shallower':         {'conv_channels': D['conv_channels'][:-1]},
        'deeper':            {'convs_per_block': D.get('convs_per_block', 1) + 1},
        'kernel5':           {'kernel_size': 5, 'padding': 2},
        'no_padding':        {'padding': 0},
        'stride2_no_pool':   {'stride': 2, 'pool': None},
        'avg_pool':          {'pool': 'avg'},
        'hidden32':          {'hidden_dims': (32,)},
        'hidden512':         {'hidden_dims': (512,)},
        'toggle_batchnorm':  {'batch_norm': not D.get('batch_norm', False)},
        'loss_hinge':        {'loss': 'hinge'},
        'loss_mse':          {'loss': 'mse'},
        'sgd':               {'optimizer': 'sgd', 'learning_rate': 0.01},
    }

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='MNIST', choices=METHODS)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--variants', nargs='*', default=None)
    args = p.parse_args()

    loader = Dataset_Loader(args.dataset, '')
    loader.dataset_source_folder_path = 'data/stage_3_data/'
    loader.dataset_source_file_name = args.dataset
    data = loader.load()          

    Cls = METHODS[args.dataset]
    all_v = get_variants(Cls.DEFAULTS)
    names = args.variants or list(all_v)

    out_path = f'result/stage_3_result/ablation_{args.dataset}.csv'
    if not os.path.exists(out_path):
        with open(out_path, 'w', newline='') as f:
            csv.writer(f).writerow(['config', 'accuracy', 'precision', 'recall', 'f1', 'params', 'sec_per_epoch'])
    for name in names:
        print(f'\n===== {name} =====')
        torch.manual_seed(2); np.random.seed(2)
        try:
            m = Cls(**{**all_v[name], 'max_epoch': args.epochs})
        except RuntimeError as e:
            print('invalid config (feature map too small):', e)
            continue
        m.data = data
        t0 = time.time(); out = m.run(); sec = (time.time() - t0) / args.epochs
        ev = Evaluate_Metrics('m', ''); ev.data = out; s = ev.evaluate()
        n_params = sum(p.numel() for p in m.parameters())
        with open(f'result/stage_3_result/ablation_{args.dataset}_{name}_pred.pkl', 'wb') as pf:
            pickle.dump({'pred_y': out['pred_y'], 'true_y': out['true_y'], 'history': out['history']}, pf)
        row = [name, round(s['accuracy'], 4), round(s['precision'], 4), round(s['recall'], 4),
               round(s['f1'], 4), n_params, round(sec, 1)]
        print('RESULT:', row)
        with open(out_path, 'a', newline='') as f:  
            csv.writer(f).writerow(row)