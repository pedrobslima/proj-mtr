from utils import *
from MSVR import MSVR
from joblib import load as jload, dump as jdump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import optuna

#self.seed = randint(0, 4294967295)#952811309
#print(self.seed)

class ExpSetup():
    def __init__(self, directory:str, random_state:int, train_models:int=None):
        '''Class made for seting up the experiments, mostly to save on
        memory space while gathering data for the assessors to train
        
        - directory: The data directory/folder inside the folder "data". 
        It's also the codename given to the experiment
        - random_state: The seed given to all methods used in this object, that use a random starting state

        - self.train_models: (VERY IMPORTANT!) Dictates how the Kolmogorov-Arnold Network will be trained:
            - 0: Won't be trained, and instead use the saved parameters and weights for this experiment, saved on the kan_params folder

            - 1: Will be trained, but won't search for the best parameters, using the saved params on kan_params/{directory}.json. 
            This will also (re)write a new weitghts (.pt) file for this experiment.

            - 2: Wil be trained and use GridSearchCV to define the best hyperparameters. This will (re)write a new 
            parameters file (.json) and a new weights file (.pt) for this experiment.
        '''
        
        self.directory = directory
        self.seed = random_state
        self.train_models = train_models

        {'heart': self._heartData,
         'churn': self._churnData,
         'covid': self._covidData,
         'bug': self._bugData
         }[self.directory]()

        self._trainBaseModels()

    def saveData(self):
        pass

    def assessorData(self, y_target:bool=False):
        '''Returns (in order):
        - Xa_train
        - Xa_test
        - Xa_test_norm
        - ya_train
        - ya_test
        - pred_corrls (train set)
        - pred_corrls2 (test set)
        - performs
        - ya_train_tgt (if y_target==True)
        - ya_test_tgt  (if y_target==True)
        '''
        return_out = [self.Xa_train, self.Xb_test_norm,
                self.Xa_test, self.Xa_test_norm,
                self.ya_train, self.ya_test,
                self.pred_corrls, self.pred_corrls2,
                self.performs]
        
        if(y_target):
            return_out = return_out + [self.ya_train_tgt, self.ya_test_tgt]

        return return_out

    def _split_data(self, df:pd.DataFrame, target:str='target', 
                    base_assessor_split:float=0.3, base_test_size:float=0.5):
        
        # (For consistency, all datasets will have the same name for the target variable
        df = df.rename(columns={target: 'target'}) 
        cols = df.drop(columns=['target']).columns

        temp = df[df.target==1]
        train_pos, test_pos = train_test_split(temp, test_size=base_assessor_split, shuffle=True, random_state=self.seed)

        temp = df[df.target==0]
        train_neg, test_neg = train_test_split(temp, test_size=base_assessor_split, shuffle=True, random_state=self.seed)

        self.Xb_train, self.Xb_test, self.yb_train, self.yb_test = train_test_split(pd.concat([train_pos[cols], train_neg[cols]], ignore_index=True), # X
                                                                    pd.concat([train_pos['target'], train_neg['target']], ignore_index=True), # y
                                                                    test_size=base_test_size, shuffle=True, random_state=self.seed)

        self.Xa_test = pd.concat([test_pos[cols], test_neg[cols]], ignore_index=True)
        self.ya_test_tgt = pd.concat([test_pos['target'], test_neg['target']], ignore_index=True)

    def _heartData(self):
        cols = ['age', 'sex', 'chest_pain', 'blood_pressure',
        'cholestoral', 'blood_sugar', 'eletro_results',
        'max_heart_rate', 'exer_angina', 'oldpeak',
        'slope_peak', 'major_vessels', 'thal']

        df = pd.read_csv(f'data/{self.directory}/heart.dat', names=cols+['target'], sep=' ')

        df['target'] = df['target'].apply(lambda x: 1 if x==2 else 0)

        print(f'Nº de instâncias: {len(df)}')
        
        self._split_data(df)

        scaler = StandardScaler()

        self.Xb_train_norm = scaler.fit_transform(self.Xb_train, self.yb_train)
        self.Xb_test_norm = scaler.transform(self.Xb_test)
        self.Xa_test_norm = scaler.transform(self.Xa_test)

    def _bugData(self):
        df = pd.read_csv(f'data/{self.directory}/bug dataset.csv', header=0)
        print('Nº de instâncias:', len(df))

        df['class'] = df['class'].replace({'NO_BUG':0, 'BUG':1})

        self._split_data(df, "class")

        # Normalizing
        scaler = StandardScaler()

        # n_refact only has 3 unique values, but it is supposed to be continuous
        self.Xb_train_norm = scaler.fit_transform(self.Xb_train, self.yb_train)

        # Normalização nos conjuntos de validação e teste, com base nos dados de treinamento
        self.Xb_test_norm = scaler.transform(self.Xb_test)

        self.Xa_test_norm = scaler.transform(self.Xa_test)

        # Balanceamento no conjunto de treinamento
        o_sampler = RandomOverSampler(random_state=self.seed)

        yb_train_norm = self.yb_train.copy()
        self.Xb_train, self.yb_train = o_sampler.fit_resample(self.Xb_train, self.yb_train)
        self.Xb_train_norm, _ = o_sampler.fit_resample(self.Xb_train_norm, yb_train_norm)

    def _churnData(self):

        df = pd.read_csv(f'data/{self.directory}/customer_churn_telecom_services.csv', header=0)
        print('Nº de instâncias:', len(df))
        #df.head()

        # Quantiades de cada valor único por coluna
        continuous_cols = []
        cat_cols = []

        for col in df.drop(columns=['Churn']).columns:
            unique_values = df[col].value_counts()
            if(len(unique_values) <= 4):
                #print(unique_values, '\n')
                #df[col] = df[col].astype('category')
                cat_cols.append(col)
            else:
                continuous_cols.append(col)

        df['TotalCharges'] = df['TotalCharges'].fillna(0)

        # Alterando colunas categóricas binárias para int

        #df_train1['gender'] = df_train1['gender'].apply(lambda x: 0 if x=='Male' else 1)

        rdict = {'gender': {'Male': 0, 'Female': 1},
                'Partner': {'No': 0, 'Yes': 1},
                'Dependents': {'No': 0, 'Yes': 1},
                'PhoneService': {'No': 0, 'Yes': 1},
                'PaperlessBilling': {'No': 0, 'Yes': 1},
                'Churn': {'No': 0, 'Yes': 1},
                }

        # Alterando colunas que são parcialmente dummy
        # Exp.: OnlineSecurity: ("No internet service", "No", "Yes") -> (0, 1, 2)

        rdict['MultipleLines'] = {'No phone service': 0, 'No': 1, 'Yes': 2}

        cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']

        for col in cols:
            rdict[col] = {'No internet service': 0, 'No': 1, 'Yes': 2}

        # Alterando colunas não-dummy

        rdict['InternetService'] = {'No': 0, 'DSL': 1, 'Fiber optic': 2}

        rdict['Contract'] = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}

        rdict['PaymentMethod'] = {'Credit card (automatic)': 0, 'Bank transfer (automatic)': 1,
                                'Mailed check': 2, 'Electronic check': 3}

        df = df.replace(rdict)

        self._split_data(df, target="Churn")
        #Xa_test_norm = scaler.transform(Xa_test)

        # Normalização baseada no conjunto de treinamento
        scaler1 = MinMaxScaler()

        self.Xb_train_norm = self.Xb_train.copy()
        self.Xb_train_norm.loc[:,continuous_cols] = scaler1.fit_transform(self.Xb_train_norm.loc[:,continuous_cols], self.yb_train)

        # Normalização nos conjuntos de validação e teste, com base nos dados de treinamento
        self.Xb_test_norm = self.Xb_test.copy()
        self.Xb_test_norm.loc[:,continuous_cols] = scaler1.transform(self.Xb_test_norm.loc[:,continuous_cols])

        self.Xa_test_norm = self.Xa_test.copy()
        self.Xa_test_norm.loc[:,continuous_cols] = scaler1.transform(self.Xa_test_norm.loc[:,continuous_cols])

        #Xb_train_norm.head()

        # Balanceamento no conjunto de treinamento
        o_sampler = RandomOverSampler(random_state=self.seed)

        yb_train_norm = self.yb_train.copy()
        self.Xb_train, self.yb_train = o_sampler.fit_resample(self.Xb_train, self.yb_train)
        self.Xb_train_norm, _ = o_sampler.fit_resample(self.Xb_train_norm, yb_train_norm) #yb_train == yb_train_norm

    def _covidData(self):

        df = pd.read_csv(f'data/{self.directory}/hosp1_v8 (1).csv') # 526 exemplos
        df2 = pd.read_csv(f'data/{self.directory}/hospital2 (2).csv').drop(columns=['creatino.fosfoquinase.cpk.plasma.ck',
                                                                        'troponina.i.plasma.troponina.i']) # 134 exemplos
        df = pd.concat([df, df2])
        print('Nº de instâncias:', len(df))
        #df.head()

        #for col in df.columns:
        #    bp = df[[col]].boxplot()
            #plt.title(col)
            #plt.show(bp)

        # Removing dastric outliers
        df = df[(df['tpg.alt.tgp']<200) & (df['tgo.ast.tgo']<200) & (df['sodio.sodio']<155) & (df['dimero.d.plasma.dimeros.d.quantitativo']<7500)]

        self._split_data(df, target='severity')

        scaler = StandardScaler()

        self.Xb_train_norm = scaler.fit_transform(self.Xb_train, self.yb_train)
        self.Xb_test_norm = scaler.transform(self.Xb_test)

        self.Xa_test_norm = scaler.transform(self.Xa_test)


    def _registerPerformance(self, train_set:list|tuple, test_set:list|tuple, model, model_name:str):
            assert (len(train_set)==2) & (len(test_set)==2)

            print(f'[REGISTERED: {model_name.upper()}]')

            # train_set
            pred = model.predict(train_set[0])

            pred_proba = model.predict_proba(train_set[0])
            minprob = [1-max(prob) for prob in pred_proba]

            metrics = compute_performance_metrics(train_set[1], pred, pred_proba[:,1])

            print('[Training]')
            print_metrics_summary(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5])
            print('{metric:<18}{value:.4f}'.format(metric="1-maxProb média:", value=np.mean(minprob)))
            print('{metric:<18}{value:.4f}'.format(metric='1-maxProb dp:', value=np.std(minprob)))
            self.performs[model_name] = {'train': {'1-maxprob':minprob, 'accuracy':metrics[0],
                                        'recall':metrics[1],
                                        'precision':metrics[2], 'f1':metrics[3],
                                        'auroc':metrics[4], 'aupr':metrics[5]}}
            # test_set
            pred = model.predict(test_set[0])
            pred_proba = model.predict_proba(test_set[0])
            minprob = [1-max(prob) for prob in pred_proba]

            metrics = compute_performance_metrics(test_set[1], pred, pred_proba[:,1])

            print('\n[Testing]')
            print_metrics_summary(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5])
            print('{metric:<18}{value:.4f}'.format(metric="1-maxProb média:", value=np.mean(minprob)))
            print('{metric:<18}{value:.4f}\n'.format(metric='1-maxProb dp:', value=np.std(minprob)))
            self.performs[model_name]['test'] = {'1-maxprob':minprob, 'accuracy':metrics[0],
                                        'recall':metrics[1],
                                        'precision':metrics[2], 'f1':metrics[3],
                                        'auroc':metrics[4], 'aupr':metrics[5]}


    def _trainBaseModels(self):
        """# Base Models"""

        self.performs = {}
        models_error = pd.DataFrame()
        models = []

        """### Decision Tree"""

        #dtree = DecisionTreeClassifier(random_state=self.seed)#.fit(self.Xb_train, self.yb_train)

        dt_params = {'max_depth':[None,10,20,30],
                    'criterion':['gini','entropy','log_loss'],
                    'min_samples_split':[2,5,10,20,40,60],
                    'min_samples_leaf':[1,2,10,30]}

        dtree = self._trainIndModel(DecisionTreeClassifier(random_state=self.seed), 'dtree', dt_params)

        self._registerPerformance((self.Xb_train, self.yb_train), (self.Xb_test, self.yb_test), dtree, 'dtree')
        models_error['dtree'] = 1 - dtree.predict_proba(self.Xb_test)[np.arange(len(self.Xb_test)), self.yb_test]
        models_error.head()

        models.append(dtree)

        """### Stochastic Gradient Descent
        (SGD)
        """

        sgd_params = {
            'loss': ['log_loss', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [1e-4, 1e-3, 1e-2]
        }

        sgd = self._trainIndModel(SGDClassifier(random_state=self.seed), 'sgd', sgd_params, norm_data=True)

        self._registerPerformance((self.Xb_train_norm, self.yb_train), (self.Xb_test_norm, self.yb_test), sgd, 'sgd')
        models_error['sgd'] = 1 - sgd.predict_proba(self.Xb_test_norm)[np.arange(len(self.Xb_test_norm)), self.yb_test]
        models.append(sgd)

        """### Logistic Regression"""

        lr_params = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }

        lr = self._trainIndModel(LogisticRegression(random_state=self.seed), 'lr', lr_params, norm_data=True)

        self._registerPerformance((self.Xb_train_norm, self.yb_train), (self.Xb_test_norm, self.yb_test), lr, 'lr')
        models_error['lr'] = 1 - lr.predict_proba(self.Xb_test_norm)[np.arange(len(self.Xb_test_norm)), self.yb_test]
        models.append(lr)

        """### K-Nearest Neighbors

        (KNN)
        """

        #knn = KNeighborsClassifier(n_jobs=-1).fit(self.Xb_train_norm, self.yb_train)

        knn_params = {
            'n_neighbors': list(range(3,21)),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        knn = self._trainIndModel(KNeighborsClassifier(), 'knn', knn_params, norm_data=True)

        self._registerPerformance((self.Xb_train_norm, self.yb_train), (self.Xb_test_norm, self.yb_test), knn, 'knn')
        models_error['knn'] = 1 - knn.predict_proba(self.Xb_test_norm)[np.arange(len(self.Xb_test_norm)), self.yb_test]
        models.append(knn)

        """### Support Vector Machine - Linear
        (SVM)
        """

        #svm_linear = SVC(kernel='linear', probability=True, random_state=self.seed).fit(self.Xb_train_norm, self.yb_train)

        svm_linear_params = {
            'C': [0.1, 1, 10]
        }

        svm_linear = self._trainIndModel(SVC(kernel='linear', probability=True, random_state=self.seed), 
                                         'svm_linear', svm_linear_params, norm_data=True)

        self._registerPerformance((self.Xb_train_norm, self.yb_train), (self.Xb_test_norm, self.yb_test), svm_linear , 'svm_linear')
        models_error['svm_linear'] = 1 - svm_linear.predict_proba(self.Xb_test_norm)[np.arange(len(self.Xb_test_norm)), self.yb_test]
        models.append(svm_linear)

        """### Support Vector Machine - Polynomial
        (SVM)
        """

        #svm_poly = SVC(kernel='poly', probability=True, random_state=self.seed).fit(self.Xb_train_norm, self.yb_train)

        svm_poly_params = {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
        }

        svm_poly = self._trainIndModel(SVC(kernel='poly', probability=True, random_state=self.seed), 
                                       'svm_poly', svm_poly_params, norm_data=True)

        self._registerPerformance((self.Xb_train_norm, self.yb_train), (self.Xb_test_norm, self.yb_test), svm_poly, 'svm_poly')
        models_error['svm_poly'] = 1 - svm_poly.predict_proba(self.Xb_test_norm)[np.arange(len(self.Xb_test_norm)), self.yb_test]
        models.append(svm_poly)

        """### Support Vector Machine - RBF
        (SVM)
        """

        #svm_rbf = SVC(kernel='rbf', probability=True, random_state=self.seed).fit(self.Xb_train_norm, self.yb_train)

        svm_rbf_params = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }

        svm_rbf = self._trainIndModel(SVC(kernel='rbf', probability=True, random_state=self.seed), 
                                      'svm_rbf', svm_rbf_params, norm_data=True)

        self._registerPerformance((self.Xb_train_norm, self.yb_train), (self.Xb_test_norm, self.yb_test), svm_rbf, 'svm_rbf')
        models_error['svm_rbf'] = 1 - svm_rbf.predict_proba(self.Xb_test_norm)[np.arange(len(self.Xb_test_norm)), self.yb_test]
        models.append(svm_rbf)

        """### Multi-Layer Perceptron
        (MLP)
        """

        #mlp = MLPClassifier(max_iter=1000, n_iter_no_change=20, random_state=self.seed).fit(self.Xb_train_norm, self.yb_train)

        mlp_params = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001]
        }

        mlp = self._trainIndModel(MLPClassifier(max_iter=1000, n_iter_no_change=20, random_state=self.seed), 
                                  'mlp', mlp_params, norm_data=True)

        self._registerPerformance((self.Xb_train_norm, self.yb_train), (self.Xb_test_norm, self.yb_test), mlp, 'mlp')
        models_error['mlp'] = 1 - mlp.predict_proba(self.Xb_test_norm)[np.arange(len(self.Xb_test_norm)), self.yb_test]
        models.append(mlp)

        """### Kolmogorov-Arnold Network
        (KAN)
        """

        if(self.train_models is None):
            self.train_models = int(input('Train KAN? (0-No / 1-Yes / 2-Yes+HPO): '))

        '''dataset = {'train_input': self.Xb_train_norm,
                'train_label': self.yb_train,
                'test_input': self.Xb_test_norm,
                'test_label': self.yb_test}'''

        INP_SIZE = self.Xb_train_norm.shape[1]
        kan_params = {'width': [[INP_SIZE,2,2,2], [INP_SIZE,3,2], [INP_SIZE,20,2]],
                    'grid': [3, 6, 8],
                    'lr': [0.0001, 0.001, 0.01],
                    'lamb': [0.005, 0.1]}

        if(self.train_models==2):
            best_score = 0
            for w in kan_params['width']:
                for g in kan_params['grid']:
                    for lr in kan_params['lr']:
                        for l in kan_params['lamb']:
                            score = cross_val_score(KANClassifier(w, grid=g, lr=lr, lamb=l, random_state=self.seed, grid_eps=1, steps=15), X=self.Xb_train_norm, y=self.yb_train, cv=10, error_score=0).mean()
                            if(score > best_score):
                                best_score = score
                                best_width = w
                                best_grid = g
                                best_lr = lr
                                best_lamb = l
            kanet = KANClassifier(best_width, grid=best_grid, lr=best_lr, lamb=best_lamb, random_state=self.seed, grid_eps=1, steps=15).fit(self.Xb_train_norm, self.yb_train)
            with open(f'models/{self.directory}/kan_params.json', 'w') as f:
                json.dump({'width': best_width, 'grid': best_grid, 'lr': best_lr, 'lamb': best_lamb}, f)
                #joblib.dump(kanet.state_dict(), 'kan_params/{self.directory}.pkl')
            save(kanet.state_dict(), f'models/{self.directory}/kan.pt')
        else:
            with open(f'models/{self.directory}/kan_params.json', 'r') as f:
                kan_dict = json.load(f)
            best_width = kan_dict['width']
            best_grid = kan_dict['grid']
            best_lr = kan_dict['lr']
            best_lamb = kan_dict['lamb']

            kanet = KANClassifier(best_width, grid=best_grid, lr=best_lr, lamb=best_lamb, random_state=self.seed, grid_eps=1, steps=15)

            if(self.train_models==0):
                kanet.load_state_dict(load(f'models/{self.directory}/kan.pt', weights_only=True))
            elif(self.train_models==1):
                kanet.fit(self.Xb_train_norm, self.yb_train)
                save(kanet.state_dict(), f'models/{self.directory}/kan.pt')

        #print('width: {},\ngrid: {},\nlr: {},\nlamb: {}'.format(best_width, best_grid, best_lr, best_lamb))

        self._registerPerformance((self.Xb_train_norm, self.yb_train), (self.Xb_test_norm, self.yb_test), kanet, 'kan')
        models_error['kan'] = 1 - kanet.predict_proba(self.Xb_test_norm)[np.arange(len(self.Xb_test_norm)), self.yb_test]
        models.append(kanet)

        """### Random Forest"""

        #rforest = RandomForestClassifier(random_state=self.seed, n_jobs=-1).fit(self.Xb_train, self.yb_train)

        parameters = {'max_depth':[None,10,20,30],
                    'criterion':['gini','entropy','log_loss'],
                    'min_samples_split':[2,5,10,20,40,60],
                    'min_samples_leaf':[1,2,10,30],
                    'n_estimators':[20,50,100,150]}

        rforest = self._trainIndModel(RandomForestClassifier(random_state=self.seed), 'rforest', parameters)

        self._registerPerformance((self.Xb_train, self.yb_train), (self.Xb_test, self.yb_test), rforest, 'rforest')
        models_error['rforest'] = 1 - rforest.predict_proba(self.Xb_test)[np.arange(len(self.Xb_test)), self.yb_test]
        models.append(rforest)

        """### Gradient Boosting"""

        #gb = GradientBoostingClassifier(random_state=self.seed).fit(self.Xb_train, self.yb_train)

        gb_params = {'max_depth':[None,10,20,30],
                    'criterion':['friedman_mse', 'squared_error'],
                    'min_samples_split':[2,5,10,20,40,60],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators':[20,50,100,150]}

        gb = self._trainIndModel(GradientBoostingClassifier(random_state=self.seed), 'gb', gb_params)

        self._registerPerformance((self.Xb_train, self.yb_train), (self.Xb_test, self.yb_test), gb, 'gb')
        models_error['gb'] = 1 - gb.predict_proba(self.Xb_test)[np.arange(len(self.Xb_test)), self.yb_test]
        models.append(gb)

        """### AdaBoost"""

        #adab = AdaBoostClassifier(random_state=self.seed).fit(self.Xb_train, self.yb_train)

        adab_params = {'learning_rate': [0.01, 0.05, 0.1, 1],
                    'n_estimators':[20,50,100,150]}

        adab = self._trainIndModel(AdaBoostClassifier(random_state=self.seed), 'adab', adab_params)

        self._registerPerformance((self.Xb_train, self.yb_train), (self.Xb_test, self.yb_test), adab, 'adab')
        models_error['adab'] = 1 - adab.predict_proba(self.Xb_test)[np.arange(len(self.Xb_test)), self.yb_test]
        models.append(adab)

        """### XGBoost"""

        #xgb = XGBClassifier(random_state=self.seed).fit(self.Xb_train, self.yb_train)

        xgb_params = {
            'n_estimators': [20,50,100,150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [None,5,10,20,30],
            'gamma': [None, 0.25, 0.5]
        }

        xgb = self._trainIndModel(XGBClassifier(random_state=self.seed), 'xgb', xgb_params)

        self._registerPerformance((self.Xb_train, self.yb_train), (self.Xb_test, self.yb_test), xgb, 'xgb')
        models_error['xgb'] = 1 - xgb.predict_proba(self.Xb_test)[np.arange(len(self.Xb_test)), self.yb_test] #xgb.predict(self.Xb_test) != self.yb_test
        models.append(xgb)

        """## Análise de Correlações"""

        #groups = {}
        model_names = list(self.performs.keys())

        self.pred_corrls = pd.DataFrame(columns=models_error.columns, index=models_error.columns, dtype=float)

        self.ya_train = pd.DataFrame(columns=model_names, index=self.yb_test.index)

        for i in range(len(models)):

            if(model_names[i] in ['dtree', 'rforest', 'gb', 'adab', 'xgb']):
                y_pred = models[i].predict_proba(self.Xb_test)
            else:
                y_pred = models[i].predict_proba(self.Xb_test_norm)

            self.ya_train.loc[:,model_names[i]] = 1 - y_pred[np.arange(len(y_pred)), self.yb_test]

        self.pred_corrls = models_error.corr('kendall')

        # Assessors

        self.Xa_train = self.Xb_test.copy()
        self.ya_train_tgt = self.yb_test
        self.ya_train = models_error.copy()

        #self.ya_train.head()

        self.ya_test = pd.DataFrame()

        for i in range(len(models)):
            if(model_names[i] in ['dtree', 'rforest', 'gb', 'adab', 'xgb']):
                y_pred = models[i].predict_proba(self.Xa_test)
            else:
                y_pred = models[i].predict_proba(self.Xa_test_norm)

            self.ya_test.loc[:,model_names[i]] = 1 - y_pred[np.arange(len(y_pred)), self.ya_test_tgt]

        self.pred_corrls2 = self.ya_test.corr('kendall')

    def _trainIndModel(self, mbase, mname:str, mparams:dict, norm_data:bool=False):
        if(self.train_models==2):
            n_jobs = 1 if isinstance(mbase, XGBClassifier) else -1
            g_search = GridSearchCV(mbase, mparams, n_jobs=n_jobs, cv=10, verbose=0)
            if(norm_data):
                g_search = g_search.fit(self.Xb_train_norm, self.yb_train)
            else:
                g_search = g_search.fit(self.Xb_train, self.yb_train)
            new_model = g_search.best_estimator_
            jdump(new_model, f'models/{self.directory}/{mname}.pkl')
        else:
            with open(f'models/{self.directory}/{mname}.pkl', 'rb') as f:
                base_model = jload(f)
            if(self.train_models):
                new_model = mbase.set_params(**base_model.get_params())
                jdump(new_model, f'models/{self.directory}/{mname}.pkl')
            else:
                new_model = base_model
        return new_model

class OptunaStudy():
    def __init__(self, x, y, num_trials, random_state):
        self.x = x
        self.y = y
        self.num_trials = num_trials
        self.model_class = None
        self.params = None
        self.best_trial = None
        self.best_params = None
        self.seed = random_state

    def search(self, model_class, **params):
        self.model_class = model_class
        self.params = params
        
        study = optuna.create_study(direction='maximize')

        if(self.model_class==RegressorChain or self.model_class==MultiOutputRegressor):
            study.optimize(self._compound_obj, n_trials=self.num_trials)

        elif(model_class==MSVR):
            self.c_range = np.asarray(range(25, 500, 25)) / 100
            study.optimize(self._msvr_obj, n_trials=self.num_trials)

        elif(model_class==RandomForestRegressor):
            study.optimize(self._rf_obj, n_trials=self.num_trials)

        self.best_trial = study.best_trial
        self.best_params = study.best_params
        
        print(self.best_trial)

    def _compound_obj(self, trial:optuna.trial._trial.Trial):
        # 2. Suggest values for the hyperparameters using a trial object.
        n_estimators = trial.suggest_int('n_estimators', 80, 160, log=True)
        criterion = trial.suggest_categorical('criterion', ['squared_error', 'poisson', 'friedman_mse'])
        max_depth = trial.suggest_int('max_depth', 10, 35, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 50, log=True)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16, log=True)

        reg = self.model_class(RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, 
                                criterion=criterion, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, random_state=self.seed))#.fit(Xa_train, ya_train)
        
        if(len(self.params)>0):
            reg.set_params(**self.params)

        scores = cross_val_score(reg, self.x, self.y, n_jobs=-1, cv=5, scoring="neg_mean_squared_error")
        #score, _ = ks_score(y_val1, clf.predict_proba(X_val1))
        
        return np.mean(scores)
    
    def _msvr_obj(self, trial:optuna.trial._trial.Trial):
        # 2. Suggest values for the hyperparameters using a trial object.
        C = trial.suggest_categorical('C', self.c_range)
        gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])
        tol = trial.suggest_float('tol', 0.001, 0.5, log=True)
        degree = trial.suggest_int('degree', 2, 8)
        epsilon = trial.suggest_float('epsilon', 0.1, 0.8, log=True)
        

        reg = self.model_class(C=C, gamma=gamma, tol=tol, degree=degree, epsilon=epsilon)#.fit(Xa_train, ya_train)
        
        if(len(self.params)>0):
            reg.set_params(**self.params)

        scores = cross_val_score(reg, self.x, self.y, n_jobs=-1, cv=5, scoring="neg_mean_squared_error")
        #score, _ = ks_score(y_val1, clf.predict_proba(X_val1))
        
        return np.mean(scores)
    
    def _rf_obj(self, trial:optuna.trial._trial.Trial):
        # 2. Suggest values for the hyperparameters using a trial object.
        n_estimators = trial.suggest_int('n_estimators', 80, 160, log=True)
        criterion = trial.suggest_categorical('criterion', ['squared_error', 'poisson', 'friedman_mse'])
        max_depth = trial.suggest_int('max_depth', 10, 35, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 50, log=True)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16, log=True)

        reg = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, 
                                criterion=criterion, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, random_state=self.seed)#.fit(Xa_train, ya_train)
        
        if(len(self.params)>0):
            reg.set_params(**self.params)

        scores = cross_val_score(reg, self.x, self.y, n_jobs=-1, cv=5, scoring="neg_mean_squared_error")
        
        return np.mean(scores)

class GroupedAssessor():
    '''Group of multi-target assessors, making predictions similarly to a comitee'''
    def __init__(self, estimators):
        self.assessors = estimators

    def fit(self, X, y, groups:list=None):
        self.cols = y.columns
        self.out_freq = {o: 0 for o in self.cols}
        if(groups is None):
            self.groups = [self.cols]*len(self.assessors)
        else:
            if(len(groups) != len(self.assessors)):
                raise ValueError('O número de grupos e de modelos deve ser igual')
            self.groups = groups

        for i, g in enumerate(self.groups):
            try:
                self.assessors[i].fit(X, y[g])
            except:
                self.assessors[i].fit(X, y[g].values)
            for m in g:
                self.out_freq[m] += 1
        return self

    def predict(self, X):
        len_x = X.shape[0]
        y_pred = pd.DataFrame(0, columns=self.cols, index=range(len_x), dtype=float)
        for i, g in enumerate(self.groups):
            y_pred.loc[:,g] = y_pred.loc[:,g] + self.assessors[i].predict(X).reshape(len_x,len(g))
        y_pred = y_pred / self.out_freq
        return np.asarray(y_pred)