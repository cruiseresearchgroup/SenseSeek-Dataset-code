import os
import pandas as pd
from datetime import datetime
import numpy as np
import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_validate

from sklearn.metrics import accuracy_score, make_scorer, f1_score, precision_score, recall_score
from sklearn.utils import shuffle

SAVE_FILE = 1

# the hyper-parameters are taken by gridsearch CV. The codes are in the train-params.py file 
models10 = {
'EEG': Pipeline(steps=[('scaling', StandardScaler()),
                ('reduce_dim', SelectKBest(k=30)),
                ('classify', SVC(C=5, degree=2, gamma=0.1, probability=True))]),
'MOTION': Pipeline(steps=[('scaling', StandardScaler()),
                   ('reduce_dim', NeighborhoodComponentsAnalysis(n_components=3)),
                ('classify', SVC(C=1, degree=2, gamma=0.001, probability=True))]),
'EDA': Pipeline(steps=[('scaling', StandardScaler()),
               ('reduce_dim',SelectKBest(k=30)),#
                ('classify', SVC(C=5, degree=2, gamma=0.1, probability=True))]),
'PUPIL': Pipeline(steps=[('scaling', StandardScaler()),
                 ('reduce_dim', NeighborhoodComponentsAnalysis(n_components=3)),
                ('classify', SVC(C=1, degree=2, gamma=0.001, probability=True))]),
'GAZE': Pipeline(steps=[('scaling', StandardScaler()),
                ('reduce_dim', NeighborhoodComponentsAnalysis(n_components=3)),
                ('classify', SVC(C=5, degree=2, gamma=0.001, probability=True))])
}



models21 = {
    'EEG': Pipeline(steps=[('scaling', StandardScaler()),
                ('reduce_dim', SelectKBest(k=25,score_func=mutual_info_classif)),
                ('classify', SVC(C=5, degree=2, gamma=0.1, probability=True))]),
'MOTION': Pipeline(steps=[('scaling', StandardScaler()),
                           ('reduce_dim', NeighborhoodComponentsAnalysis(n_components=3)),
                ('classify', SVC(C=1, degree=2, gamma=0.001, probability=True))]),
'EDA': Pipeline(steps=[('scaling', StandardScaler()),
                ('reduce_dim',SelectKBest(k=30,score_func=mutual_info_classif)),#
                ('classify', SVC(C=5, degree=2, gamma=0.1, probability=True))]),
      'PUPIL': Pipeline(steps=[('scaling', StandardScaler()),
                ('reduce_dim', SelectKBest(k=15,score_func=mutual_info_classif)),
                ('classify', SVC(C=5, degree=2, gamma=0.1, probability=True))]),
    'GAZE': Pipeline(steps=[('scaling', StandardScaler()),
                ('reduce_dim', NeighborhoodComponentsAnalysis(n_components=3)),
                ('classify', SVC(C=5, degree=2, gamma=0.001, probability=True))])
}


models42 = {
    'EEG': Pipeline(steps=[('scaling', StandardScaler()),
                 ('reduce_dim', SelectKBest(k=30,score_func=mutual_info_classif)),
                ('classify', SVC(C=5, degree=4, gamma=0.1, probability=True))]),
    'MOTION': Pipeline(steps=[('scaling', StandardScaler()),
                    ('reduce_dim', None),
                    ('classify', SVC(C=5, degree=2, gamma=0.01, probability=True))]),
    'EDA': Pipeline(steps=[('scaling', StandardScaler()),
                    ('reduce_dim',LinearDiscriminantAnalysis(n_components=2)),#
                    ('classify', SVC(C=5, degree=2, gamma=0.1, probability=True))]),
    'PUPIL': Pipeline(steps=[('scaling', StandardScaler()),
                ('reduce_dim', NeighborhoodComponentsAnalysis(n_components=3)),
                ('classify', SVC(C=1, degree=2, gamma=0.001, probability=True))]),
    'GAZE': Pipeline(steps=[('scaling', StandardScaler()),
               ('reduce_dim', NeighborhoodComponentsAnalysis(n_components=2)),
                ('classify', SVC(C=5, degree=2, gamma=0.001, probability=True))])
}


### common properties
datafiles = {
    'EEG': 'trainning data/EEG',
        'MOTION': 'trainning data/MOTION', #motion + e4 acc
    'PUPIL': 'trainning data/PUPIL',
    'GAZE': 'trainning data/GAZE',
        'EDA': 'trainning data/EDA',}

data_types = ['EEG', 'MOTION','PUPIL', 'EDA', 'GAZE' ]
info_cols =  ['task',  'stage',]
id_col = ['PID']

target_name = 'stage'
target_set = ['IN', 'QF', 'TYPE', 'SPEAK', 'LISTEN','READ',]

if __name__ == '__main__':
    scoring = {"ROC": 'roc_auc_ovr', "Accuracy": make_scorer(accuracy_score), "F1-macro": make_scorer(f1_score, average='macro', zero_division=0),
              "Precision": make_scorer(precision_score, average="macro", zero_division=0), 
              "Recall": make_scorer(recall_score, average="macro", zero_division=0),}

    current_ver = datetime.now().strftime("(v3)%d_%m_%Y-%I_%M_%p")
    save_dir = f'{current_ver}'
    try:
        os.mkdir(os.path.join('models', save_dir))
    except:
        pass
        
    for (tw, models) in [('1-0', models10), ('4-2', models42), ('2-1', models21)]:     
        #### read data
        for key, file in datafiles.items():   
            print('Processing', key)
            df = pd.read_csv(f'{file}({len(target_set)} {tw}).csv')
            df = df.drop(columns=['Topic'])
            
            features_cols = df.columns.values
            features_cols = [c for c in features_cols if c not in info_cols + id_col]
            print('Extracted ', len(features_cols), 'features in total', features_cols[0])
            
            # pids = df.PID.unique()
                
            groups = df.PID.values
            print(df.PID.value_counts())
            
            X = df[features_cols].values
            y = df[target_name].values
            X, y, groups_shuffled = shuffle(X, y, groups, random_state=12345)

            print(np.unique(groups_shuffled, return_counts=True))
            
            pipe = models[key]
            logo = LeaveOneGroupOut()
            outer_cv = cross_validate(pipe, X, y, \
                                      n_jobs=10, cv=logo, groups=groups_shuffled, scoring=scoring, \
                                      return_train_score=True, return_estimator=True, return_indices=True)
            y_train = []
            train_group = []
            for n, idx in enumerate(outer_cv['indices']['train']):
                y_train.append(y[idx])
                train_group.append(groups_shuffled[idx])
            
            y_pred = []
            y_true = []
            y_group = []
            for n, idx in enumerate(outer_cv['indices']['test']):
                y_true.append(y[idx])
                clf = outer_cv['estimator'][n]
                pred = clf.predict(X[idx])
                y_pred.append(pred)
                y_group.append(groups_shuffled[idx])
            
            outer_cv.update({'y_pred': y_pred, 'y_true': y_true, 'test_group': y_group, 'y_train': y_train, 'train_group': train_group})
            if SAVE_FILE == 1:
                save_file = os.path.join('models', save_dir, f'SVM({len(target_set)})-{tw}-{key}-{datetime.now().strftime("%d_%m_%Y-%I_%M_%p")}.pkl')
                pickle.dump(outer_cv, open(save_file, 'wb'))
                print('SAVED MODEL')
                    
            print(key, 'DONE')