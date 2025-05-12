
# ------ IMPORTS ------
# Basic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
import networkx as nx
import json
from torch import save, load
from sklearn.model_selection import GridSearchCV

# Base Models (+ ST assessor)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR # kernels: 'linear', 'poly' e 'rbf'
from sklearn.neural_network import MLPClassifier
from kan import KANClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# Multi-Target Assessors
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

# Performance eval
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, root_mean_squared_error as rmse
#from sklearn.inspection import PartialDependenceDisplay as pdp
#from sklearn.inspection import permutation_importance
from itertools import product

def getAssrParams(estimator, estimator_params, random_state=42, mtr_seed=True, estimator_var='base_estimator'):
    mtr_params = {estimator_var: []}
    
    if('random_state' in estimator().get_params().keys()):
        estimator_params['random_state'] = [random_state]
        if(mtr_seed):
            mtr_params['random_state'] = [random_state]
    
    params_keys = list(estimator_params.keys())
    lists_params = product(*estimator_params.values())
    
    for params in lists_params:
        estimator_params = {}

        for i in range(len(params)):
            estimator_params[params_keys[i]] = params[i]

        mtr_params[estimator_var].append(estimator(**estimator_params))

    return mtr_params

# ------ FUNCTIONS ------
def extract_final_losses(history):
    """Função para extrair o melhor loss de treino e validação.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    Dicionário contendo o melhor loss de treino e de validação baseado 
    no menor loss de validação.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    idx_min_val_loss = np.argmin(val_loss)
    return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}

def plot_training_error_curves(history):
    """Função para plotar as curvas de erro do treinamento da rede neural.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    A função gera o gráfico do treino da rede e retorna None.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Train')
    ax.plot(val_loss, label='Validation')
    ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (MSE)')
    ax.legend()
    plt.show()

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class, zero_division=0)
    precision = precision_score(y, y_pred_class, zero_division=0)
    f1 = f1_score(y, y_pred_class, zero_division=0)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)
    return performance_metrics

def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))

def nmse(y_true, y_pred, sample_weight=None, 
         multioutput="uniform_average"):
    mse_error = mse(y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values")
    variance = np.var(np.asarray(y_true), axis=0)
    
    if 0 in variance.flatten():
        raise ValueError("A variância dos valores reais é zero; não é possível normalizar.")
    
    output_errors = mse_error / variance
    #y_pred = np.asarray(y_pred)
    #y_true = np.asarray(y_true)

    #output_errors = np.average((y_pred - y_true)**2 / (y_true - np.mean(y_true, axis=0).reshape(1,-1))**2, 
    #                           axis=0, weights=sample_weight)
    
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)

def nmae(y_true, y_pred, sample_weight=None, 
         multioutput="uniform_average"):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    #output_errors = np.average(np.abs(y_pred - y_true) / np.abs(y_true - np.mean(y_true, axis=0).reshape(1,-1)),
    #                           axis=0, weights=sample_weight)

    output_errors = np.average(np.abs(y_pred - y_true), axis=0, weights=sample_weight) / np.average(np.abs(y_true - np.mean(y_true, axis=0).reshape(1,-1)), axis=0)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)

def reltArray(y):
    return y - np.average(y, axis=0)

def aCC(y_true, y_pred):
    '''average Correlation Coefficient'''
    arrTrue = reltArray(np.asarray(y_true))
    arrPred = reltArray(np.asarray(y_pred))
    output_values = np.sum(arrTrue * arrPred, axis=0) / np.sqrt(np.sum(arrTrue**2, axis=0) * np.sum(arrPred**2, axis=0))
    return np.mean(output_values)

def aRRMSE(y_true, y_pred):
    '''average Relative Root-Mean Squared Error'''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = np.sum((y_true-y_pred)**2, axis=0) / np.sum(reltArray(y_true)**2, axis=0) 
    return np.mean(np.sqrt(out))

def corrCoeff(y_true, y_pred, multioutput="uniform_average"):
    '''Correlation coefficient'''
    arrTrue = reltArray(np.asarray(y_true))
    arrPred = reltArray(np.asarray(y_pred))
    output_values = np.sum(arrTrue * arrPred, axis=0) / np.sqrt(np.sum(arrTrue**2, axis=0) * np.sum(arrPred**2, axis=0))
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_values
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return np.average(output_values, weights=multioutput)

def rrmse(y_true, y_pred, multioutput="uniform_average"):
    '''Relative root-mean squared error'''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    output_errors = np.sqrt(np.sum((y_true-y_pred)**2, axis=0) / np.sum(reltArray(y_true)**2, axis=0))
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)

def group_search(df: pd.DataFrame, min_score: int):
    G = nx.from_pandas_adjacency(df)

    # Remover arestas com peso menor que MIN_SCORE
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_score]
    G.remove_edges_from(edges_to_remove)
    cliques = list(nx.find_cliques(G))
    # Encontrar os subgrafos conectados (componentes conexos)
    #subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    subgraphs = [G.subgraph(clique).copy() for clique in cliques]
    groups = []
    out_groups = []
    # Exibir os subgrafos
    for sg in subgraphs:
        groups.append(set(sg.nodes()))

    for i in range(len(groups)):
        check = True
        j=0
        while(j < len(groups) and check):
            if(i!=j):
                check = not(groups[i]<=groups[j])
            j+=1
        if(check):
            out_groups.append(list(groups[i]))
    return out_groups

def corrlArray(df_corrl:pd.DataFrame):
    corrls_values = df_corrl.reset_index().melt(id_vars='index', var_name='model2', value_name='corr')
    corrls_values = corrls_values.rename(columns={'index': 'model1'})
    corrls_values = corrls_values[corrls_values['model1']!=corrls_values['model2']].reset_index(drop=True)
    # ordenando strings por linha
    corrls_values.loc[:,['model1','model2']] = pd.DataFrame(np.sort(corrls_values.loc[:,['model1','model2']].values, axis=1), columns=['model1','model2']) 
    return corrls_values.drop_duplicates().reset_index(drop=True)

def metricsFormat(df: pd.DataFrame, metric_name:str):
    df = df.reset_index().melt(id_vars='index', var_name='model', value_name=metric_name)
    df = df.rename(columns={'index': 'assessor_type'})

    return df

def save2figs(heatmap, histogram, directory:str):
    '''Save a Heatmap and its corresponding Histogram into 2 .eps files'''
    title1 = heatmap.get_title('center')
    code2 = ''.join([s[0].lower() for s in histogram.get_title('center').split()])
    if(title1.count(' ') > 3):
        title1 = title1.replace('Set', 'set').replace('Train', 'TRAIN').replace('Test', 'TEST').replace('P', 'p')
        code1 = ''.join([c.lower() for c in title1 if c.isupper()])
        code1 = code1[:2] + '_' + code1[2:]
    elif(title1.find('(') != -1):
        title1 = title1[title1.index('(')+1:title1.index(')')]
        code1 = 'cmd_' + ''.join([c.lower() for c in title1 if c.isupper()])
    else:
        code1 = 'cmd_base'
    code2 += code1[code1.index('_'):]
    
    heatmap.figure.savefig(f'data/{directory}/imgs/{code1}.eps', format='eps', dpi=300)
    print(f'[Saved image "{code1}.eps"]')
    histogram.figure.savefig(f'data/{directory}/imgs/{code2}.eps', format='eps', dpi=300)
    print(f'[Saved image "{code2}.eps"]')

def assr_performance(assr, X_test, y_test, X_train=None, y_train=None):

    if(not((X_train is None) or (y_train is None))):
        y_pred = assr.predict(X_train)

        print(f'''[TRAINING SET]
MSE: {mse(y_train, y_pred):.4f}
NMSE: {nmse(y_train, y_pred):.4f}
MAE: {mae(y_train, y_pred):.4f}
NMAE: {nmae(y_train, y_pred):.4f}
aRRMSE: {rrmse(y_train, y_pred):.4f}
aCC: {corrCoeff(y_train, y_pred):.4f}''')

    y_pred = assr.predict(X_test)

    mse_array = mse(y_test, y_pred, multioutput='raw_values')
    nmse_array = nmse(y_test, y_pred, multioutput='raw_values')
    mae_array = mae(y_test, y_pred, multioutput='raw_values')
    nmae_array = nmae(y_test, y_pred, multioutput='raw_values')
    rrmse_array = rrmse(y_test, y_pred, multioutput='raw_values')
    corrCoeff_array = corrCoeff(y_test, y_pred, multioutput='raw_values')
    
    print(f'''\n[TESTING SET]
MSE: {np.average(mse_array):.4f}
NMSE: {np.average(nmse_array):.4f}
MAE: {np.average(mae_array):.4f}
NMAE: {np.average(nmae_array):.4f}
aRRMSE: {np.average(rrmse_array):.4f}
aCC: {np.average(corrCoeff_array):.4f}''')
    
    return np.asarray([mse_array,
                       nmse_array,
                       mae_array,
                       nmae_array,
                       rrmse_array,
                       corrCoeff_array]).T


def correlation_matrix(corrls, show_plot:bool, save_plot:str, pred_name:str, set_name:str):
    hm = plt.figure(figsize=(20,8))

    plt.title(f'Correlation Matrix ({pred_name} {set_name} Set Predictions)', fontsize=17)
    hm.axes[0] = sns.heatmap(corrls, annot=True, ax=hm.axes[0])
    hm.axes[0].xaxis.tick_top()

    assr_corrls_values = corrlArray(corrls)['corr']

    hg = plt.figure()
    plt.title('Correlation Distribution')
    hg.axes[0] = assr_corrls_values.hist()
    
    if(show_plot):
        plt.show(hm)
        plt.show(hg)
    else:
        plt.close(hm)
        plt.close(hg)
    
    print(f'{set_name} correlation distribution:')
    print(assr_corrls_values.describe())

    if(save_plot is not None):
        save2figs(hm.axes[0], hg.axes[0], save_plot)

def correlation_pred_performance(assr, X_test, y_test, X_train=None, min_score:float=0.7,
                                 show_plot:bool=True, save_plot:str=None, name=None):
    assr_groups = {}

    if(name is None):
        name=str(assr)
        name = name[:name.find('(')]

    if(X_train is not None):
        assr_corrls = pd.DataFrame(assr.predict(X_train), 
                          columns=y_test.columns, 
                          index=X_train.index).corr('kendall')
        
        correlation_matrix(assr_corrls, show_plot, save_plot, name, 'Training')
        assr_groups['train'] = group_search(assr_corrls, min_score)

    assr_corrls = pd.DataFrame(assr.predict(X_test), 
                          columns=y_test.columns, 
                          index=X_test.index).corr('kendall')
    
    correlation_matrix(assr_corrls, show_plot, save_plot, name, 'Testing')
    assr_groups['test'] = group_search(assr_corrls, min_score)

    pred_corrls = y_test.corr('kendall')

    performance = [mse(pred_corrls, assr_corrls), 
                   mae(pred_corrls, assr_corrls), 
                   nmse(pred_corrls, assr_corrls), 
                   nmae(pred_corrls, assr_corrls), 
                   rrmse(pred_corrls, assr_corrls), 
                   corrCoeff(pred_corrls, assr_corrls)]

    diff_corrls = assr_corrls - pred_corrls

    diff_corrls_values = corrlArray(diff_corrls)['corr']

    print(F'''
Diferença entre correlações das previsões do assessor e do conjunto de teste:
MSE: {performance[0]:.4f}
NMSE: {performance[1]:.4f}
MAE: {performance[2]:.4f}
NMAE: {performance[3]:.4f}
aRRMSE: {performance[4]:.4f}
aCC: {performance[5]:.4f}''')

    hm = plt.figure(figsize=(20,8))
    #plt.title(r'$Corr_{pred(MultiRandomForest)}$ - $Corr_{true(test)}$', fontsize=21)
    plt.title('$Corr_{pred(' + name + ')}$ - $Corr_{true(test)}$', fontsize=21)
    hm.axes[0] = sns.heatmap(diff_corrls, annot=True, cmap='coolwarm', vmin=-0.5, vmax=0.5, ax=hm.axes[0])
    hm.axes[0].xaxis.tick_top()

    hg = plt.figure()
    plt.title('Correlation Difference Distribution')
    hg.axes[0] = diff_corrls_values.hist()

    if(show_plot):
        plt.show(hm)
        plt.show(hg)
    else:
        plt.close(hm)
        plt.close(hg)
        
    if(save_plot is not None):
        save2figs(hm.axes[0], hg.axes[0], save_plot)
    
    return {'results': performance, 'groups': assr_groups}