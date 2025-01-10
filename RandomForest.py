import pandas as pd  # Per la gestione dei dati
import numpy as np  # Per operazioni numeriche avanzate
import seaborn as sns  # Per creare visualizzazioni
import matplotlib.pyplot as plt  # Per la visualizzazione grafica
from sklearn.ensemble import RandomForestClassifier  # Per il modello RandomForest
from sklearn.model_selection import RandomizedSearchCV  # Per la ricerca dei parametri ottimali
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score  # Per valutare il modello
from sklearn.model_selection import train_test_split  # Per dividere i dati in set di addestramento e test
from sklearn.tree import plot_tree  # Per visualizzare gli alberi decisionali

# Definizione della classe RandomForest
class RandomForest:
    def __init__(self, df, target_column):
        """
        Il costruttore della classe prende il dataset e il nome della colonna target come input.
        """
        self.df = df  # Assegna il dataset al membro della classe
        self.target_column = target_column  # Assegna la colonna target
        self.model = RandomForestClassifier(class_weight='balanced')  # Inizializza il modello RandomForest con pesi bilanciati
        self.nrf = None  # Inizializza l'attributo per il modello dopo la ricerca dei parametri
        self.x_train, self.x_test, self.y_train, self.y_test = self._prepare_data()  # Prepara i dati

    def _prepare_data(self):
        """
        Metodo per separare le features (X) dalla target (y) e dividere i dati in set di addestramento e test.
        """
        X = self.df.drop(columns=[self.target_column])  # Rimuove la colonna target dalle feature
        y = self.df[self.target_column]  # Separa la colonna target
        return train_test_split(X, y, test_size=0.25, random_state=42)  # Divide i dati in 75% training, 25% testing

    def optimize_parameters(self):
        """
        Ottimizza i parametri del modello RandomForest utilizzando RandomizedSearchCV.
        """
        # Definizione della griglia dei parametri da esplorare
        params = {
            'criterion': ['gini', 'entropy'],  # Sceglie il criterio per la divisione dei nodi
            'min_samples_split': list(np.arange(2, 51)),  # Numero minimo di campioni richiesti per dividere un nodo
            'min_samples_leaf': list(np.arange(2, 51)),  # Numero minimo di campioni in un nodo foglia
            'max_features': ['sqrt', 'log2', None],  # Numero di feature da considerare per ogni divisione
            'n_estimators': [1000]  # Numero di alberi nel bosco
        }

        # Creazione del RandomizedSearchCV per trovare i migliori parametri
        self.nrf = RandomizedSearchCV(self.model, param_distributions=params, cv=10, n_jobs=-1, scoring='accuracy')
        self.nrf.fit(self.x_train, self.y_train)  # Allena il modello con i dati di addestramento

        # Stampa i migliori parametri e il miglior punteggio
        print("Best Parameters:", self.nrf.best_params_)
        print("Best Score:", self.nrf.best_score_)

    def evaluate_model(self):
        """
        Valuta il modello ottimizzato sui dati di addestramento e test e stampa le metriche.
        """
        nrf_best = self.nrf.best_estimator_  # Prende il miglior modello trovato da RandomizedSearchCV
        y_train_pred = nrf_best.predict(self.x_train)  # Predizione sui dati di addestramento
        y_test_pred = nrf_best.predict(self.x_test)  # Predizione sui dati di test

        # Stampa del report di classificazione per i dati di addestramento
        print("\nTraining Classification Report:")
        print(classification_report(self.y_train, y_train_pred))

        # Stampa del report di classificazione per i dati di test
        print("\nTesting Classification Report:")
        print(classification_report(self.y_test, y_test_pred))

        # Stampa delle metriche di valutazione per addestramento e test
        print("Accuracy (Train):", accuracy_score(self.y_train, y_train_pred))  # Accuracy per training
        print("Accuracy (Test):", accuracy_score(self.y_test, y_test_pred))  # Accuracy per test
        print("Precision (Train):", precision_score(self.y_train, y_train_pred))  # Precision per training
        print("Precision (Test):", precision_score(self.y_test, y_test_pred))  # Precision per test
        print("Recall (Train):", recall_score(self.y_train, y_train_pred))  # Recall per training
        print("Recall (Test):", recall_score(self.y_test, y_test_pred))  # Recall per test
        print("F1 Score (Train):", f1_score(self.y_train, y_train_pred))  # F1 Score per training
        print("F1 Score (Test):", f1_score(self.y_test, y_test_pred))  # F1 Score per test

        # Chiamata per la visualizzazione delle matrici di confusione
        self.plot_confusion_matrix(self.y_train, y_train_pred, "Training Confusion Matrix")
        self.plot_confusion_matrix(self.y_test, y_test_pred, "Testing Confusion Matrix")

        # Chiamata per la visualizzazione dell'importanza delle feature
        self.plot_feature_importance(nrf_best)

        # Chiamata per visualizzare uno degli alberi della RandomForest
        self.plot_random_forest()

    def plot_confusion_matrix(self, y_true, y_pred, title):
        """
        Metodo per visualizzare la matrice di confusione come heatmap.
        """
        cm = confusion_matrix(y_true, y_pred)  # Calcola la matrice di confusione
        plt.figure(figsize=(6, 5))  # Imposta la dimensione della figura
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])  # Crea una heatmap
        plt.title(title)  # Titolo della matrice
        plt.xlabel('Predicted Labels')  # Etichetta asse x
        plt.ylabel('True Labels')  # Etichetta asse y
        plt.show()  # Mostra il grafico

    def plot_feature_importance(self, model):
        """
        Metodo per visualizzare l'importanza delle feature nel modello RandomForest.
        """
        importances = model.feature_importances_  # Ottiene le importanze delle feature
        indices = np.argsort(importances)[::-1]  # Ordina le feature in ordine decrescente di importanza
        features = self.x_train.columns  # Ottiene i nomi delle feature

        plt.figure(figsize=(10, 6))  # Imposta la dimensione della figura
        sns.barplot(x=importances[indices], y=features[indices])  # Crea un grafico a barre
        plt.title("Feature Importance")  # Titolo del grafico
        plt.xlabel("Relative Importance")  # Etichetta asse x
        plt.ylabel("Features")  # Etichetta asse y
        plt.show()  # Mostra il grafico

    def plot_random_forest(self):
        """
        Visualizza uno degli alberi decisionali della RandomForest.
        """
        # Prende il miglior modello
        rf_best_model = self.nrf.best_estimator_

        # Seleziona il primo albero del modello RandomForest
        estimator = rf_best_model.estimators_[0]

        # Crea il grafico dell'albero
        plt.figure(figsize=(20, 10))
        plot_tree(estimator, filled=True, feature_names=self.x_train.columns, class_names=['0', '1'])
        plt.title("Decision Tree of the RandomForest")
        plt.show()

# Funzione main che avvia il flusso del programma
def main():
    # Carica il dataset da un file CSV
    ds = pd.read_csv('diabetes.csv')  # Carica il dataset chiamato 'diabetes.csv'

    # Creazione dell'oggetto RandomForest, passando il dataset e la colonna target 'Outcome'
    rf_model = RandomForest(ds, 'Outcome')

    # Ottimizzazione dei parametri del modello tramite RandomizedSearchCV
    rf_model.optimize_parameters()

    # Valutazione del modello ottimizzato
    rf_model.evaluate_model()


# Se il file viene eseguito direttamente, avvia la funzione main
if __name__ == "__main__":
    main()
