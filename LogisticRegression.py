from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    classification_report, roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LogisticRegressionModel:

    #Costruttore della classe per il modello (parametri: X (contiene caratteristiche dataset (comprese le due colonne create Age_Category e Normalized_Age),y (valori di Outcome))
    def __init__(self,X,x_te,y,y_te,train_m,test_m):
        self.X_train = X
        self.y_train = y
        self.X_test = x_te
        self.y_test = y_te
        self.t_m = train_m
        self.te_m = test_m

    def train_model(self):
        # Suddivisione in training (75%) e testing set (25%)
        #X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=20)

        self.t_m.append("Logistic Regression")
        self.te_m.append("Logistic Regression")

        #Definizione dei parametri per RandomizedSearchCV
        regressor = LogisticRegression(max_iter=1000)
        params = {'penalty': ['l1', 'l2'],'solver': ['saga', 'liblinear'],'C': list(np.arange(1, 21))}
        #Ricerca dei migliori parametri
        nreg = RandomizedSearchCV(regressor,param_distributions=params,scoring='accuracy', n_jobs=-1,cv=10,random_state=42)

        #Addestramento del modello
        nreg.fit(self.X_train, self.y_train)


        # Miglior modello
        best_model = nreg.best_estimator_
        # Calcolo delle metriche
        pred_train=nreg.predict(self.X_train)
        pred_test = nreg.predict(self.X_test)

        # Stampa il miglior parametro
        print(nreg.best_params_)
        # Stampa l'accuratezza
        print("Accuracy:", nreg.best_score_)

        # Controllo che il modello migliore non sia None
        if best_model is None:
            raise ValueError("Il modello migliore non è stato trovato.")

        # Stampa del report di classificazione per i dati di addestramento
        print("\nTraining Classification Report:")
        print(classification_report(self.y_train, pred_train))

        # Stampa del report di classificazione per i dati di test
        print("\nTesting Classification Report:")
        print(classification_report(self.y_test, pred_test))

        # Stampa delle metriche di valutazione per addestramento e test
        print("Accuracy (Train):", accuracy_score(self.y_train,  pred_train))# Accuracy per training
        self.t_m.append(f"accuracy: {accuracy_score(self.y_train,  pred_train)}")
        print("Accuracy (Test):", accuracy_score(self.y_test, pred_test))  # Accuracy per test
        self.te_m.append(f"accuracy: {accuracy_score(self.y_test, pred_test)}")
        print("Precision (Train):", precision_score(self.y_train,  pred_train))  # Precision per training
        self.t_m.append(f"precision: {precision_score(self.y_train, pred_train)}")
        print("Precision (Test):", precision_score(self.y_test, pred_test))  # Precision per test
        self.te_m.append(f"precision: {precision_score(self.y_test, pred_test)}")
        print("Recall (Train):", recall_score(self.y_train,  pred_train))  # Recall per training
        self.t_m.append(f"recall: {recall_score(self.y_train, pred_train)}")
        print("Recall (Test):", recall_score(self.y_test, pred_test))  # Recall per test
        self.te_m.append(f"recall: {recall_score(self.y_test, pred_test)}")
        print("F1 Score (Train):", f1_score(self.y_train,  pred_train))  # F1 Score per training
        self.t_m.append(f"f1_score: {f1_score(self.y_train, pred_train)}")
        print("F1 Score (Test):", f1_score(self.y_test, pred_test))  # F1 Score per test
        self.te_m.append(f" f1_score: {f1_score(self.y_test, pred_test)}")

        # Chiamata per la visualizzazione delle matrici di confusione
        self.plot_confusion_matrix(self.y_train, pred_train, "Training Confusion Matrix")
        self.plot_confusion_matrix(self.y_test, pred_test, "Testing Confusion Matrix")

        # Calcolo della curva ROC per test set
        if hasattr(best_model, 'predict_proba'):
            fpr, tpr, _ = roc_curve(self.y_test, best_model.predict_proba(self.X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            self.plot_roc_curve(fpr, tpr, roc_auc)
        else:
            fpr, tpr, roc_auc = None, None, None

    def plot_confusion_matrix(self, y_true, y_pred, title):

        # Metodo per visualizzare la matrice di confusione come heatmap.

        cm = confusion_matrix(y_true, y_pred)  # Calcola la matrice di confusione
        plt.figure(figsize=(6, 5))  # Imposta la dimensione della figura
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])  # Crea una heatmap
        plt.title(title)  # Titolo della matrice
        plt.xlabel('Predicted Labels')  # Etichetta asse x
        plt.ylabel('True Labels')  # Etichetta asse y
        plt.savefig(f'logistic_regression{title}.png', dpi=300, bbox_inches='tight')
        plt.show()  # Mostra il grafico

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        plt.figure(figsize=(10, 6))

        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'logistic_regressionROC.png', dpi=300, bbox_inches='tight')
        plt.show()
