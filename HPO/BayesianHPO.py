



def build_Bayes_search_alg(search_alg:str,param_ranges:dict):
    
    alg = None

    from ray.tune.search.bayesopt import BayesOptSearch
    
    alg = BayesOptSearch(
        param_ranges,
        metric="test_f1_score",
        mode="max",
        utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
    )    

    return alg