from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def train_and_eval(x_train,
                   y_train,
                   x_test,
                   y_test,
                   max_depth=16, n_estimators=100):
    # init clf
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                 n_streams=1,
                                 random_state = 42)
    # fit clf 
    clf.fit(x_train,y_train)
    # predict 
    y_pred = clf.predict(x_test)
    # get f1 score 
    score = f1_score(y_test.to_numpy(),y_pred.to_numpy())

    return score


    
    

def objective(trial):
    """
    Define a search space for the hyperparameters `n_estimators` and `max_depth`
    of a random forest model, then train and evaluate it using cross validation.
    """
    from DataLoader import getPreprocessedDataOnGPU
    
    x_train , y_train , x_test , y_test = getPreprocessedDataOnGPU()
    
    n_estimators = trial.suggest_int('n_estimators', 100, 300)
    max_depth = int(trial.suggest_float('max_depth', 1, 100, log=True))

    
    score = train_and_eval(x_train,y_train,x_test,y_test,
                           n_estimators=n_estimators,
                           max_depth=max_depth)
    
    return score