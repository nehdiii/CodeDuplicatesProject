from cuml.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Decicion Trees CPU HPO with naive bayes 

def train_and_eval_DT(x_train,
                   y_train,
                   x_test,
                   y_test,
                   max_depth=16,
                   criterion="gini",
                   splitter = "random",
                   random_state=21 ,  
                    ):
    # init clf
    clf = DecisionTreeClassifier(
                                max_depth=max_depth, 
                                criterion=criterion,
                                splitter=splitter,
                                random_state =random_state)
    # fit clf 
    clf.fit(x_train[:,:,0],y_train[:,0])
    # predict 
    y_pred = clf.predict(x_test[:,:,0])
    # get f1 score 
    score = f1_score(y_test[:,0],y_pred)

    return score




def objective_DT(trial):

    from DataLoader import getPreprocessedData
    
    x_train , y_train , x_test , y_test = getPreprocessedData()
    

    max_depth = int(trial.suggest_float('max_depth', 1 , 500 , log=True))
    criterion = trial.suggest_categorical("criterion",['gini', 'entropy', 'log_loss'])
    splitter = trial.suggest_categorical("splitter",['best', 'random'])
   
     
    score = train_and_eval_DT(x_train,y_train,x_test,y_test,
                           max_depth=max_depth,
                           criterion=criterion,
                           splitter=splitter)
                  
    
    return score
    
# Decicion Trees GPU optimization with naive bayes 

def train_and_eval_RF(x_train,
                   y_train,
                   x_test,
                   y_test,
                   split_criterion="gini",
                   max_depth=10, 
                   n_estimators=10,
                   bootstrap=False):
    # init clf
    clf = RandomForestClassifier(
                                 split_criterion=split_criterion,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 bootstrap = bootstrap,
                                 n_streams=1,
                                 random_state = 21)
    # fit clf 
    clf.fit(x_train,y_train)
    # predict 
    y_pred = clf.predict(x_test)
    # get f1 score 
    score = f1_score(y_test.to_numpy(),y_pred.to_numpy())
    
  
    return score



def objective_RF(trial):
    """
    Define a search space for the hyperparameters `n_estimators` and `max_depth`
    of a random forest model, then train and evaluate it using cross validation.
    """
    from DataLoader import getPreprocessedDataOnGPU
    
    x_train , y_train , x_test , y_test = getPreprocessedDataOnGPU()
    
    n_estimators = trial.suggest_int('n_estimators', 1 , 100)
    max_depth = int(trial.suggest_float('max_depth', 1 , 100, log=True))
    bootstrap = bool(trial.suggest_int('bootstrap', 0 , 1))
    #split_criterion = trial.suggest_categorical("split_criterion",['inverse_gaussian','poisson'])#, 'mse','gamma','inverse_gaussian','poisson'])
    
    score = train_and_eval_RF(x_train,y_train,x_test,y_test,
                           n_estimators=n_estimators,
                           max_depth=max_depth,
                           bootstrap=bootstrap)
    
    return score