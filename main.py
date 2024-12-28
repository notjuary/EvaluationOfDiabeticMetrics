import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]





def main():
    ds = pd.read_csv('diabetes.csv') #lettura dataset
    """#stampa l'intero dataset
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(ds) 
    ds.info()  # mostra info sul dataset
    print(ds.isnull().any()) # ricerca di valori nulli
    print(ds.duplicated().sum()) #ricerca di eventuali duplicati """

    #creazione dei grafici per le varie colonne per la distribuzione dei valori
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(ds.columns):
        row, col_position = divmod(i, 3)
        sns.boxplot(data=ds, y=col, ax=axes[row, col_position])
        plt.tight_layout()
    plt.show()

    for col in ds.columns:
        ds = remove_outliers(ds, col)
    
    
if __name__ == "__main__":
    main()