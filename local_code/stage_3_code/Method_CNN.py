# code/stage_3_code/Method_CNN.py
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from local_code.base_class.method import method


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_CNN(method, nn.Module):
    data = None

    def __init__(self, mName, mDescription, input_shape, n_classes,
                 conv_channels=(32, 64), convs_per_block=1, kernel_size=3,
                 stride=1, padding=1, pool='max', batch_norm=False,
                 hidden_dims=(128,), dropout=0.25, loss='ce', optimizer='adam',
                 learning_rate=1e-3, max_epoch=10, batch_size=64, augment=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.n_classes = n_classes
        self.loss_name, self.optimizer_name = loss, optimizer
        self.learning_rate, self.max_epoch = learning_rate, max_epoch
        self.batch_size, self.augment = batch_size, augment
        self.device = get_device()

        layers, in_c = [], input_shape[0]
        for out_c in conv_channels:
            for _ in range(convs_per_block):
                layers.append(nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ReLU(inplace=True))
                in_c = out_c
            if pool == 'max':
                layers.append(nn.MaxPool2d(2))
            elif pool == 'avg':
                layers.append(nn.AvgPool2d(2))
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, *input_shape)).numel()

        fc, d = [nn.Flatten()], flat_dim
        for h in hidden_dims:
            fc += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        fc.append(nn.Linear(d, n_classes))
        self.classifier = nn.Sequential(*fc)

    def forward(self, x):
        return self.classifier(self.features(x))

    def compute_loss(self, logits, y):
        if self.loss_name == 'ce':
            return F.cross_entropy(logits, y)
        if self.loss_name == 'ce_smooth':
            return F.cross_entropy(logits, y, label_smoothing=0.1)
        if self.loss_name == 'hinge':
            return F.multi_margin_loss(logits, y)
        if self.loss_name == 'mse':
            return F.mse_loss(F.softmax(logits, 1), F.one_hot(y, self.n_classes).float())
        raise ValueError(self.loss_name)

    def _augment(self, x):
        flip = torch.rand(x.size(0), device=x.device) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x)

    def train_model(self, X, y, X_test=None, y_test=None):
        self.to(self.device)
        if self.optimizer_name == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                            batch_size=self.batch_size, shuffle=True)
        self.history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(1, self.max_epoch + 1):
            self.train()
            total_loss, correct, n, t0 = 0.0, 0, 0, time.time()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if self.augment:
                    xb = self._augment(xb)
                logits = self(xb)
                loss = self.compute_loss(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                n += xb.size(0)

            test_acc = float((self.predict(X_test) == y_test).mean()) if X_test is not None else None
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(total_loss / n)
            self.history['train_acc'].append(correct / n)
            self.history['test_acc'].append(test_acc)
            print(f'Epoch {epoch}: loss={total_loss/n:.4f} train_acc={correct/n:.4f} '
                  f'test_acc={test_acc:.4f} ({time.time()-t0:.1f}s)')

    @torch.no_grad()
    def predict(self, X, batch_size=512):
        self.eval()
        preds = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(self.device)
            preds.append(self(xb).argmax(1).cpu())
        return torch.cat(preds).numpy()

    def run(self):
        print('method running...')
        self.train_model(self.data['train']['X'], self.data['train']['y'],
                         self.data['test']['X'], self.data['test']['y'])
        pred_y = self.predict(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': self.history}