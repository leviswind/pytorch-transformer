# -*- coding: utf-8 -*-

'''
Janurary 2018 by Wei Li
liweihfyz@sjtu.edu.cn
https://www.github.cim/leviswind/transformer-pytorch
'''

from __future__ import print_function
import codecs
import os

import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from nltk.translate.bleu_score import corpus_bleu
from AttModel import AttModel
from torch.autograd import Variable
import torch


def eval():
    # Load data
    X, Sources, Targets = load_test_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    enc_voc = len(de2idx)
    dec_voc = len(en2idx)

    # load model
    model = AttModel(hp, enc_voc, dec_voc)
    model.load_state_dict(torch.load(hp.model_dir + '/model_epoch_%02d' % hp.eval_epoch + '.pth'))
    print('Model Loaded.')
    model.eval()
    # Inference
    if not os.path.exists('results'):
        os.mkdir('results')
    with codecs.open('results/model%d.txt' % hp.eval_epoch, 'w', 'utf-8') as fout:
        list_of_refs, hypotheses = [], []
        for i in range(len(X) // hp.batch_size):
            # Get mini-batches
            x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
            sources = Sources[i * hp.batch_size: (i + 1) * hp.batch_size]
            targets = Targets[i * hp.batch_size: (i + 1) * hp.batch_size]

            # Autoregressive inference
            x_ = Variable(torch.LongTensor(x))
            preds_t = torch.LongTensor(np.zeros((hp.batch_size, hp.maxlen), np.int32))
            preds = Variable(preds_t)
            for j in range(hp.maxlen):

                _, _preds, _ = model(x_, preds)
                preds_t[:, j] = _preds.data[:, j]
                preds = Variable(preds_t.long())
            preds = preds.data.cpu().numpy()

            # Write to file
            for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                fout.write("- source: " + source + "\n")
                fout.write("- expected: " + target + "\n")
                fout.write("- got: " + got + "\n\n")
                fout.flush()

                # bleu score
                ref = target.split()
                hypothesis = got.split()
                if len(ref) > 3 and len(hypothesis) > 3:
                    list_of_refs.append([ref])
                    hypotheses.append(hypothesis)
            # Calculate bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            fout.write("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    eval()
    print('Done')



