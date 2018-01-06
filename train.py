# -*- coding: utf-8 -*-

'''
Janurary 2018 by Wei Li
liweihfyz@sjtu.edu.cn
https://www.github.cim/leviswind/transformer-pytorch
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
from data_load import get_batch_indices, load_de_vocab, load_en_vocab

from torch.autograd import Variable
import os
from AttModel import AttModel
import torch
import torch.optim as optim
from data_load import load_train_data
import time
import cPickle as pickle
from tensorboardX import SummaryWriter


def train():
    current_batches = 0
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    enc_voc = len(de2idx)
    dec_voc = len(en2idx)
    writer = SummaryWriter()
    # Load data
    X, Y = load_train_data()
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    model = AttModel(hp, enc_voc, dec_voc)
    model.train()
    model.cuda()
    torch.backends.cudnn.benchmark = True
    if not os.path.exists(hp.model_dir):
        os.makedirs(hp.model_dir)
    if hp.preload is not None and os.path.exists(hp.model_dir + '/history.pkl'):
        history = pickle.load(file(hp.model_dir + '/history.pkl', 'r'))
    else:
        history = {'current_batches': 0}
    current_batches = history['current_batches']
    optimizer = optim.Adam(model.parameters(), lr=hp.lr, betas=[0.9, 0.98], eps=1e-8)
    if hp.preload is not None and os.path.exists(hp.model_dir + '/optimizer.pth'):
        optimizer.load_state_dict(torch.load(hp.model_dir + '/optimizer.pth'))
    if hp.preload is not None and os.path.exists(hp.model_dir + '/model_epoch_%02d.pth' % hp.preload):
        model.load_state_dict(torch.load(hp.model_dir + '/model_epoch_%02d.pth' % hp.preload))

    startepoch = int(hp.preload) if hp.preload is not None else 1
    for epoch in range(startepoch, hp.num_epochs + 1):
        current_batch = 0
        for index, current_index in get_batch_indices(len(X), hp.batch_size):
            tic = time.time()
            x_batch = Variable(torch.LongTensor(X[index]).cuda())
            y_batch = Variable(torch.LongTensor(Y[index]).cuda())
            toc = time.time()
            tic_r = time.time()
            torch.cuda.synchronize()
            optimizer.zero_grad()
            loss, _, acc = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            toc_r = time.time()
            current_batches += 1
            current_batch += 1
            if current_batches % 10 == 0:
                writer.add_scalar('./loss', loss.data.cpu().numpy()[0], current_batches)
                writer.add_scalar('./acc', acc.data.cpu().numpy()[0], current_batches)
            if current_batches % 5 == 0:
                print('epoch %d, batch %d/%d, loss %f, acc %f' % (epoch, current_batch, num_batch, loss.data[0], acc.data[0]))
                print('batch loading used time %f, model forward used time %f' % (toc - tic, toc_r - tic_r))
            if current_batches % 100 == 0:
                writer.export_scalars_to_json(hp.model_dir + '/all_scalars.json')
        pickle.dump(history, file(hp.model_dir + '/history.pkl', 'w'))
        checkpoint_path = hp.model_dir + '/model_epoch_%02d' % epoch + '.pth'
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(optimizer.state_dict(), hp.model_dir + '/optimizer.pth')


if __name__ == '__main__':
    train()
