"""
    출처: https://quokkas.tistory.com/entry/pytorch%EC%97%90%EC%84%9C-EarlyStop-%EC%9D%B4%EC%9A%A9%ED%95%98%EA%B8%B0

"""


import torch
import numpy as np
import os

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, best_score=np.inf, counter=0, delta=0, path='checkpoint.pt', verbose=False):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = counter
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss > self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Early Stopping Validated')
                self.early_stop = True

        else:
            self.save_checkpoint(val_loss, model)
            self.best_score = val_loss
            self.counter = 0

        return self.best_score, self.counter, self.early_stop


    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if os.path.isfile(self.path):
            os.remove(self.path)
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)




def KFOLD_GCN(DATA, KFOLD):
    CLS_TMP = list()
    NUM_CLASSES = len(set(DATA[:, -1]))
    for i in range(NUM_CLASSES):
        clssp = []
        for j in DATA:
            if int(j[-1]) == i:
                clssp.append(j)
        clssp = np.array(clssp)
        CLS_TMP.append(clssp)

    def KFOLD_SEPARATION(sample):
        getsam = sample
        kfoldset = []
        for i in range(KFOLD):
            putnum = len(getsam) // (KFOLD - i)
            wtput = getsam[:putnum]
            kfoldset.append(wtput)
            getsam = getsam[putnum:]

        return kfoldset

    SEPARATED_SET = []
    for i in CLS_TMP:
        SEPARATED_SET.append(KFOLD_SEPARATION(i))

    FINAL_SET = []
    for i in np.transpose(SEPARATED_SET):
        FINAL_SET.append(np.vstack(i))

    return FINAL_SET









