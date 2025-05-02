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
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneGroupOut, cross_validate

from sklearn.metrics import accuracy_score, make_scorer, f1_score, precision_score, recall_score
from sklearn.utils import shuffle

SAVE_FILE = 1


### common properties
datafiles = {
    'EEG': 'trainning data/EEG',
        'MOTION': 'trainning data/MOTION', #motion + e4 acc
    'PUPIL': 'trainning data/PUPIL',
    'GAZE': 'trainning data/GAZE',
        'EDA': 'trainning data/EDA',
}


info_cols =  ['task',  'stage', 'Topic',]
id_col = ['PID']

target_name = 'stage'
target_set = ['IN', 'QF', 'TYPE', 'SPEAK', 'LISTEN','READ',]


if __name__ == '__main__':

    for tw in ['4-2', '2-1', '1-0']:
        scoring = {"ROC": 'roc_auc_ovr', "Accuracy": make_scorer(accuracy_score), "F1-macro": make_scorer(f1_score, average='macro', zero_division=0),
                  "Precision": make_scorer(precision_score, average="macro", zero_division=0), 
                  "Recall": make_scorer(recall_score, average="macro", zero_division=0),}
        
        current_ver = datetime.now().strftime("(v3-param)%d_%m_%Y-%I_%M_%p")
        save_dir = f'({tw}){current_ver}'
        try:
            os.mkdir(os.path.join('models', save_dir))
        except:
            pass
    
        
        #### read data
        for key, file in datafiles.items():   
            print('Processing', key)
            #######################
            ### prepare data     ##
            #######################
            df = pd.read_csv(f'{file}({len(target_set)} {tw}).csv')
            
            features_cols = df.columns.values
            features_cols = [c for c in features_cols if c not in info_cols + id_col]
            print('Extracted ', len(features_cols), 'features in total', features_cols[0])
            # pids = df.PID.unique()
            groups = df.PID.values
            
            X = df[features_cols].values
            y = df[target_name].values
            X, y, groups_shuffled = shuffle(X, y, groups, random_state=12345)
    
            #######################
            ### prepare trainning     ##
            #######################
            N_FEATURES_OPTIONS = []
            for n in [15, 20, 25, 30, 35, 40, 45, 50]:
                if n < len(features_cols):
                    N_FEATURES_OPTIONS.append(n)
            if len(N_FEATURES_OPTIONS) == 0:
                N_FEATURES_OPTIONS.append(len(features_cols))
                    
            C_OPTIONS = [0.1, 1, 5]
            GAMMA_OPTIONS = [0.1,0.01,0.001]
            DEGREE_OPTIONS = [2,3,4,5]
            
            param_grid = [
                {
                    "reduce_dim": [LinearDiscriminantAnalysis(), NeighborhoodComponentsAnalysis()],
                    "reduce_dim__n_components": [2,3],
                    "classify__C": C_OPTIONS,
                    "classify__gamma": GAMMA_OPTIONS,
                    "classify__degree": DEGREE_OPTIONS,
                },
                {
                    "reduce_dim": [SelectKBest(mutual_info_classif), SelectKBest(f_classif)],
                    "reduce_dim__k": N_FEATURES_OPTIONS,
                    "classify__C": C_OPTIONS,
                    "classify__gamma": GAMMA_OPTIONS,
                    "classify__degree": DEGREE_OPTIONS,
                },
            ]
            reducer_labels = ["LDA", "NCA", "KBest(mutual_info_classif)", "KBest(f_classif)"]
            
            pipe = Pipeline(
                [
                    ("scaling", StandardScaler()),
                    # the reduce_dim stage is populated by the param_grid
                    ("reduce_dim", "passthrough"),
                    ("classify", SVC(probability=True, kernel='rbf')),
                ]
            )
            
            #######################
            ### train model     ##
            #######################
            
            inner_cv = KFold(n_splits=2, shuffle=True, random_state=123456)
            logo = LeaveOneGroupOut()
            
            clf = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=inner_cv, \
                               n_jobs=10, refit="F1-macro", scoring=scoring, return_train_score=True)
            
            outer_cv = cross_validate(clf, X, y, \
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
            
            outer_cv.update({'y_pred': y_pred, 'y_true': y_true, 'group': y_group})
            if SAVE_FILE == 1:
                save_file = os.path.join('models', save_dir, f'SVM({len(target_set)})-{tw}-{key}-{datetime.now().strftime("%d_%m_%Y-%I_%M_%p")}.pkl')
                pickle.dump(outer_cv, open(save_file, 'wb'))
                print('SAVED MODEL')
                
            print(key, 'DONE')
