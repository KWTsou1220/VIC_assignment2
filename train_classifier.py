from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np
import time
import pickle
import xgboost

import os


def train_classifier(data_path='./Dataset', save_path='./Models'):
    
    if os.path.isfile(os.path.join(save_path, 'xgb.sav')):
        print('Classifier exists.')
        return None
    
    x_train = np.load(os.path.join(data_path, 'x_train.npy'))
    t_train = np.load(os.path.join(data_path, 't_train.npy'))

    start = time.time()
    xgb = xgboost.XGBClassifier(n_estimators=2000, criterion='entropy', max_depth=3)
    xgb.fit(x_train, t_train)
    end = time.time()

    print('Training time: ', end-start)
    print('Training accuracy: ', xgb.score(x_train, t_train))
    y_train_xgb = xgb.predict(x_train)
    print('F1 score: ', f1_score(y_pred=y_train_xgb, y_true=t_train))

    pickle.dump(xgb, open(os.path.join(save_path, 'xgb.sav'), 'wb'))
