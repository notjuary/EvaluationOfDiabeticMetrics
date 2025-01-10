from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
class LogisticRegressionModel:

    #Costruttore della classe per il modello (parametri: X (contiene caratteristiche dataset (comprese le due colonne create Age_Category e Normalized_Age),y (valori di Outcome))
    def __init__(self,X,y):
        self.X = X
        self.y = y


    def train_model(self):
        # Suddivisione in training (75%) e testing set (25%)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=20)

        #Definizione dei parametri per RandomizedSearchCV
        regressor = LogisticRegression(max_iter=1000)
        params = {'penalty': ['l1', 'l2'],
                  'solver': ['saga', 'liblinear'],
                  'C': list(np.arange(1, 21))
        }


        #Ricerca dei migliori parametri
        nreg = RandomizedSearchCV(
            regressor,
            param_distributions=params,
            scoring='accuracy', n_jobs=-1,
            cv=10,
            random_state=42)

        #Addestramento del modello
        nreg.fit(X_train, y_train)
        regressor.fit(X_train, y_train)

        # Miglior modello
        best_model = nreg.best_estimator_
        # Calcolo delle metriche
        pred_train=regressor.predict(X_train)
        pred_test = regressor.predict(X_test)

        # Controllo che il modello migliore non sia None
        if best_model is None:
            raise ValueError("Il modello migliore non è stato trovato.")

        metrics_train = {
                'Accuracy': accuracy_score(y_train, pred_train),
                'F1-Score': f1_score(y_train, pred_train, average='weighted'),
                'Precision': precision_score(y_train, pred_train, average='weighted'),
                'Recall': recall_score(y_train, pred_train, average='weighted'),
        }

        metrics_test = {
                'Accuracy': accuracy_score(y_test, pred_test),
                'F1-Score': f1_score(y_test, pred_test, average='weighted'),
                'Precision': precision_score(y_test, pred_test, average='weighted'),
                'Recall': recall_score(y_test, pred_test, average='weighted'),
        }
        # Calcolo della matrice di confusione per test set
        cm_test = confusion_matrix(y_test, pred_test)

        # Calcolo della curva ROC per test set
        if hasattr(best_model, 'predict_proba'):
            fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None

        #Restituzione dei risultati
        return best_model,X_train,y_train,X_test,y_test,metrics_train,metrics_test,cm_test,fpr,tpr,roc_auc
