import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from DecisionTree import DecisionTree
from LogisticRegression import LogisticRegressionModel
from RandomForest import RandomForest


#Rimozione outliers
def remove_outliers(df, column):
    #Calcolo del primo quartile(Q1)
    Q1 = df[column].quantile(0.25)
    # Calcolo del terzo quartile(Q1)
    Q3 = df[column].quantile(0.75)
    # Calcolo dell'intervallo interquartile(IQR)
    IQR = Q3 - Q1
    # Calcolo del limite inferiore
    lower_bound = Q1 - 1.5 * IQR
    # Calcolo del limite superiore
    upper_bound = Q3 + 1.5 * IQR
    # Restituzione dei dati che rientrano nei limiti
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def dataset_analysis(ds):
    # stampa l'intero dataset
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(ds)
    ds.info()  # mostra info sul dataset
    print(ds.isnull().any())  # ricerca di valori nulli
    print(ds.duplicated().sum())  # ricerca di eventuali duplicati

def show_graphics(ds):
    # creazione dei grafici per le varie colonne per la distribuzione dei valori
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(ds.columns):
        row, col_position = divmod(i, 3)
        sns.boxplot(data=ds, y=col, ax=axes[row, col_position])
        plt.tight_layout()
    plt.savefig(f'box_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def find_min_max(ds):

    #calcola il minimo e il massimo dell'età
    max = ds['Age'].max()
    min = ds['Age'].min()
    print(f"Età minima: {min} Età massima: {max}")

    #definizione intervalli età per la suddivisione
    bins = [20, 36, 51, 66, float('inf')] #split sull'età
    labels = ['Young Adults', 'Middle-Aged Adults', 'Older Adults', 'Seniors'] #definizione delle etichette in base all'età
    ds['Age_Category'] = pd.cut(ds['Age'], bins=bins, labels=labels, right=True) #si assegnano le etichette all'età
    le = LabelEncoder() #conversione delle età in numeri
    ds['Age_Category'] = le.fit_transform(ds['Age_Category'])
    ds['Normalized_Age'] = (ds['Age'] - ds['Age'].min()) / (ds['Age'].max() - ds['Age'].min()) #viene creata una nuova colonna con le età normalizzate

    ds.info() #stampa del dataset

def models_evalutation_metrics(m,s):
    # Liste separate per i modelli e le loro metriche
    models = []
    metrics = []
    # Separare i dati in liste
    current_model = None
    print(f"-------------------- {s} --------------------")
    for item in m:
        if ':' not in item:
            if current_model:
                models.append(current_model)
                metrics.append(current_metrics)
            current_model = item
            current_metrics = []
        else:
            current_metrics.append(item)

    # Aggiungere l'ultimo modello e le sue metriche
    models.append(current_model)
    metrics.append(current_metrics)

    # Visualizzare i risultati
    for model, metric_list in zip(models, metrics):
        print(f"{model}")
        for metric in metric_list:
            print(f"  - {metric}")
        print()  # per separare i modelli

def feature_engineering(df):
    tdf = df.copy() #copia del dataset
    x = tdf.drop(columns='Outcome') #assegno ad x i valori di ogni colonna del dataset tranne quelli della colonna outcome
    y = tdf['Outcome'] #assegno ad y i valori della colonna outcome
    mi = mutual_info_classif(x, y) #calcolo la dipendenza tra x e y
    mi_df = pd.DataFrame({'Feature': x.columns, 'Mutual Information': mi}) #creo di un nuovo dataframe con due colonne
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True) #ordino il dataframe

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(df.corr(), annot=True, cmap='magma', ax=ax[0])
    ax[0].set_title('Correlation')
    sns.barplot(x='Mutual Information', y='Feature', data=mi_df, ax=ax[1])
    ax[1].set_title('Mutual Information')
    plt.suptitle('Correlation and Mutual Information before adding a new Feature')
    plt.tight_layout()
    plt.show()


def main():
    ds = pd.read_csv('diabetes.csv') #lettura dataset

    dataset_analysis(ds) # chiama la funzione per analizzare il dataset

    show_graphics(ds) #chiama la funzione per visualizzare i grafici sui dati del dataset

    for col in ds.columns:
        ds = remove_outliers(ds, col)

    feature_engineering(ds) #chiama la funzione per stamapre la matrice di correlazione con il grafico a barre

    find_min_max(ds)

    feature_engineering(ds) #chiama la funzione per stamapre la matrice di correlazione con il grafico a barre


    x = ds.drop(columns='Outcome')
    y = ds['Outcome']
    x_t, x_te, y_t, y_te = train_test_split(x, y, test_size=0.25, random_state=20)
    test_metrics=[]
    train_metrics=[]

    # chiamata classe LogisticRegression
    print("-------------------- Logistic Regression --------------------")
    logRegModel = LogisticRegressionModel(x_t, x_te, y_t, y_te,train_metrics,test_metrics)
    logRegModel.train_model()

    #chiama la classe DecisionTree
    print("-------------------- Decision Tree --------------------")
    dt = DecisionTree(x_t, x_te, y_t, y_te,train_metrics,test_metrics)
    dt.decision()

    #Chiamata alla RandomForest
    print("-------------------- Random Forest --------------------")
    rf = RandomForest(ds,'Outcome',x_t, x_te, y_t, y_te,train_metrics,test_metrics) #creo il randomforest
    rf.optimize_parameters() #ottimizzazione dei parametri

    models_evalutation_metrics(train_metrics,"Traning Metrics")
    models_evalutation_metrics(test_metrics,"Test Metrics")

if __name__ == "__main__":
    main()