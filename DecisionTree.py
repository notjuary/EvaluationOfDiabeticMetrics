import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, auc, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier,plot_tree # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from DecisionTreePhoto import DecisionTreePhoto
import seaborn as sns


class DecisionTree:
    def __init__(self,x_t,x_te,y_t,y_te,train_m,test_m):
        self.x_t, self.x_te, self.y_t, self.y_te, self.t_m,self.te_m = x_t, x_te, y_t, y_te, train_m, test_m

    def decision(self):

        self.t_m.append("Decision Tree")
        self.te_m.append("Decision Tree")

        # Creo l'oggetto di tipo DecisionTreeClassifier bilanciato sui pesi
        clf = DecisionTreeClassifier(class_weight='balanced')
        # Addestro il Decision Tree con i dati di x_t e y_t
        clf = clf.fit(self.x_t, self.y_t)
        # Il modello predice le etichette per i dati di test
        y_pred = clf.predict(self.x_te)
        # Calcolo dell'accuratezza
        print("Accuracy:", metrics.accuracy_score(self.y_te, y_pred))

        # Creo un oggetto DecisionTreePhoto per stampare l'immagine del Decision Tree
        dtp = DecisionTreePhoto()
        # Chiamo il metodo per stampare l'immagine
        dtp.photo(clf,self.x_t.columns)

        # Creo un oggetto DecisionTree bilanciato sui pesi
        dt = DecisionTreeClassifier(class_weight='balanced')
        # Calcolo il valore di ogni nodo in modo da capire la propria posizione
        path = dt.cost_complexity_pruning_path(self.x_t, self.y_t)
        alphas = path.ccp_alphas
        params = {'ccp_alpha': alphas}
        #Creo un oggetto di tipo RandomizedSearchCv per cercare i parametri migliori per l'albero
        ndt = RandomizedSearchCV(dt, param_distributions=params, cv=10, n_jobs=-1, scoring='accuracy', n_iter=5)
        # Addestra il modello sui dati forniti
        ndt.fit(self.x_t, self.y_t)
        # Stampa il miglior parametro
        print(ndt.best_params_)
        # Stampa l'accuratezza
        print("Accuracy:", ndt.best_score_)
        # Salvo il miglior parametro
        best_alpha = ndt.best_params_['ccp_alpha']

        # Chiamo il metodo per stampare l'immagine
        dtp.photo(ndt.best_estimator_,self.x_t.columns)

        # Creo l'oggetto di tipo DecisionTreeClassifier bilanciato sui pesi
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        # Addestro il Decision Tree con i dati di x_t e y_t
        clf = clf.fit(self.x_t, self.y_t)
        # Il modello predice le etichette per i dati di test
        y_pred = clf.predict(self.x_te)
        # Calcolo dell'accuratezza
        print("Accuracy:", metrics.accuracy_score(self.y_te, y_pred))

        # Chiamo il metodo per stampare l'immagine
        dtp.photo(clf, self.x_t.columns)

        # Creo l'oggetto di tipo DecisionTreeClassifier utilizzando il valore ottimale
        dt = DecisionTreeClassifier(ccp_alpha=best_alpha)
        # Si definiscono i parametri
        params = {'criterion': ['gini', 'entropy'], 'min_samples_split': list(np.arange(2, 51)),'min_samples_leaf': list(np.arange(2, 51))}
        # Creo un oggetto di tipo RandomizedSearchCv per cercare i parametri migliori per l'albero
        ndt = RandomizedSearchCV(dt, param_distributions=params, cv=10, n_jobs=-1, scoring='accuracy')
        # Addestra il modello sui dati forniti
        ndt.fit(self.x_t, self.y_t)
        # Stampa il miglior parametro
        print(ndt.best_params_)
        # Stampa l'accuratezza
        print("Accuracy:", ndt.best_score_)

        # Chiamo il metodo per stampare l'immagine
        dtp.photo(ndt.best_estimator_,self.x_t.columns)

        #chiamo il metodo per la stampa delle metriche
        self.model_valutation(ndt)

    def model_valutation(self,ndt):
        ndt=ndt.best_estimator_ # Prende il miglior modello trovato da RandomizedSearchCV
        y_train_pred = ndt.predict(self.x_t)  # Predizione sui dati di addestramento
        y_test_pred = ndt.predict(self.x_te)  # Predizione sui dati di test

        # Stampa del report di classificazione per i dati di addestramento
        print("\nTraining Classification Report:")
        print(classification_report(self.y_t, y_train_pred))

        # Stampa del report di classificazione per i dati di test
        print("\nTesting Classification Report:")
        print(classification_report(self.y_te, y_test_pred))

        # Stampa delle metriche di valutazione per addestramento e test
        print("Accuracy (Train):", accuracy_score(self.y_t, y_train_pred))  # Accuracy per training
        self.t_m.append(f"accuracy: {accuracy_score(self.y_t,  y_train_pred)}")
        print("Accuracy (Test):", accuracy_score(self.y_te, y_test_pred))  # Accuracy per test
        self.te_m.append(f"accuracy: {accuracy_score(self.y_te,  y_test_pred)}")
        print("Precision (Train):", precision_score(self.y_t, y_train_pred))  # Precision per training
        self.t_m.append(f"precision: {precision_score(self.y_t, y_train_pred)}")
        print("Precision (Test):", precision_score(self.y_te, y_test_pred))  # Precision per test
        self.te_m.append(f"precision: {precision_score(self.y_te, y_test_pred)}")
        print("Recall (Train):", recall_score(self.y_t, y_train_pred))  # Recall per training
        self.t_m.append(f"recall: {recall_score(self.y_t, y_train_pred)}")
        print("Recall (Test):", recall_score(self.y_te, y_test_pred))  # Recall per test
        self.te_m.append(f"recall: {recall_score(self.y_te, y_test_pred)}")
        print("F1 Score (Train):", f1_score(self.y_t, y_train_pred))  # F1 Score per training
        self.t_m.append(f"f1_score: {f1_score(self.y_t, y_train_pred)}")
        print("F1 Score (Test):", f1_score(self.y_te, y_test_pred))  # F1 Score per test
        self.te_m.append(f"f1_score: {f1_score(self.y_te, y_test_pred)}")

        # Chiamata per la visualizzazione delle matrici di confusione
        self.plot_confusion_matrix(self.y_t, y_train_pred, "Training Confusion Matrix")
        self.plot_confusion_matrix(self.y_te, y_test_pred, "Testing Confusion Matrix")

        # Calcolo della curva ROC per test set
        if hasattr(ndt, 'predict_proba'):
            fpr, tpr, _ = roc_curve(self.y_te, ndt.predict_proba(self.x_te)[:, 1])
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
        plt.savefig(f'decision_tree{title}.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(f'decision_treeROC.png', dpi=300, bbox_inches='tight')
        plt.show()