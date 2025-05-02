import os
import pandas as pd
import numpy as np
from datetime import datetime
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
from sklearn.ensemble import VotingClassifier

SAVE_FILE = 1



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

datafile = 'trainning data/'
data_types = ['EEG', 'MOTION','PUPIL', 'EDA', 'GAZE' ]
info_cols =  ['task',  'stage',]
id_col = ['PID']

target_name = 'stage'
target_set = ['IN', 'QF', 'TYPE', 'SPEAK', 'LISTEN','READ',]

combines = [#26
    ('MOTION', 'PUPIL'), ('MOTION', 'EDA'), ('EEG', 'GAZE'), ('PUPIL', 'EDA'), ('PUPIL', 'GAZE'), ('EDA', 'GAZE'), ('EEG', 'PUPIL'), ('MOTION', 'GAZE'), ('EEG', 'MOTION'), ('EEG', 'EDA'), #10
('MOTION', 'EDA', 'GAZE'), ('EEG', 'MOTION', 'EDA'), ('PUPIL', 'EDA', 'GAZE'), ('MOTION', 'PUPIL', 'EDA'), ('EEG', 'MOTION', 'GAZE'), ('MOTION', 'PUPIL', 'GAZE'), ('EEG', 'PUPIL', 'EDA'), ('EEG', 'MOTION', 'PUPIL'), ('EEG', 'PUPIL', 'GAZE'), ('EEG', 'EDA', 'GAZE'),#10
('EEG', 'MOTION', 'PUPIL', 'GAZE'), ('EEG', 'MOTION', 'EDA', 'GAZE'), ('MOTION', 'PUPIL', 'EDA', 'GAZE'), ('EEG', 'PUPIL', 'EDA', 'GAZE'), ('EEG', 'MOTION', 'PUPIL', 'EDA'), #5
('EEG', 'MOTION','PUPIL', 'EDA', 'GAZE') #1
 ]


def read_data(file):
    df = pd.read_csv(file)
    df = df.drop(columns=['Topic'])
    features_cols = [c for c in df.columns.values if c not in info_cols + id_col]
    return df, features_cols

if __name__ == '__main__':
    scoring = {"ROC": 'roc_auc_ovr', "Accuracy": make_scorer(accuracy_score), "F1-macro": make_scorer(f1_score, average='macro', zero_division=0),
              "Precision": make_scorer(precision_score, average="macro", zero_division=0), 
              "Recall": make_scorer(recall_score, average="macro", zero_division=0),}

    current_ver = datetime.now().strftime("(fusion)%d_%m_%Y-%I_%M_%p")
    save_dir = f'{current_ver}'
    try:
        os.mkdir(os.path.join('models', save_dir))
    except:
        pass
        
    for (tw, models) in [('1-0', models10), ('4-2', models42), ('2-1', models21)]:     
        
        for comb in combines:
        
            key = '+'.join(comb)
            current_features = np.array([])
            props = {}
            #combine feature files
            df = None
            for typ in comb:
                dd, features = read_data(f'{datafile}{typ}({len(target_set)} {tw}).csv')
                props.update({typ: (models[typ], features)})
                current_features = np.append(current_features, features)
                
                if df is None: 
                    df = dd
                else:
                    df = df.merge(dd, on=id_col+info_cols, how='inner')
            
            
            groups = df.PID.values
            logo = LeaveOneGroupOut()
            
            X = df[current_features].values
            y = df[target_name].values
            X, y, groups_shuffled = shuffle(X, y, groups, random_state=12345)

            pipe = Pipeline(steps=[('scaling', StandardScaler()),
                    ('classify', VotingClassifier(estimators=[(typ, val[0]) for (typ, val) in props.items()], voting='soft', weights=None))
            ])
            
            outer_cv = cross_validate(pipe, X, y, \
                                      n_jobs=10, cv=logo, groups=groups_shuffled, scoring=scoring, \
                                      return_train_score=True, return_estimator=True, return_indices=True)
            y_pred = []
            y_true = []
            y_group = []
            for n, idx in enumerate(outer_cv['indices']['test']):
                y_true.append(y[idx])
                clf = outer_cv['estimator'][n]
                pred = clf.predict(X[idx])
                y_pred.append(pred)
                y_group.append(groups_shuffled[idx])
            
            outer_cv.update({'y_pred': y_pred, 'y_true': y_true, 'group': y_group, 'datasize': len(X)})
            if SAVE_FILE == 1:
                save_file = os.path.join('models', save_dir, f'FUSION({len(target_set)})-{tw}-{key}-{datetime.now().strftime("%d_%m_%Y-%I_%M_%p")}.pkl')
                pickle.dump(outer_cv, open(save_file, 'wb'))
                print('SAVED MODEL')
                    
            print(key, 'DONE')
