import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.impute import SimpleImputer

# Überprüfe DAten ob Bedingungen für Anwendungen von ML Alogirthmen erfüllt sind

##############
#%%# Data ####
##############

data = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Working Data/Data_Ready_Tidy_2025-01-01_18-07-21.csv")
data['restaurants_price_range'] = data['restaurants_price_range'].astype('category')

###################
#%%# Check KNN ####
###################


def check_knn(data):
    # 1. Überprüfen auf numerische Spalten
    print("Numerische Spalten:")
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    print(numerical_cols)
    
    # 2. Skalierung der numerischen Daten (für KNN erforderlich)
    print("\nSkalierung der numerischen Daten (Standardabweichung vor und nach der Skalierung):")
    scaler = StandardScaler()
    numerical_data = data[numerical_cols]
    scaled_data = scaler.fit_transform(numerical_data)
    print("Standardabweichung vor Skalierung:", numerical_data.std())
    print("Standardabweichung nach Skalierung:", scaled_data.std(axis=0))

    # 3. Überprüfen auf Ausreißer (für KNN besonders wichtig)
    print("\nAusreißer (Z-Score > 3) in numerischen Spalten:")
    z_scores = np.abs(zscore(numerical_data))
    outliers = (z_scores > 3).sum(axis=0)
    print(outliers)

# Beispielaufruf für KNN
check_knn(data)


###########################
#%%# Check Naive Bayes ####
###########################

def check_naive_bayes(data):
    # 1. Überprüfen auf numerische und kategoriale Spalten
    print("Kategoriale Spalten (für Naive Bayes relevant):")
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    print(categorical_cols)

    # 2. Imputation fehlender Werte (für Naive Bayes wichtig)
    print("\nImputation fehlender Werte (nur für numerische Spalten):")
    imputer = SimpleImputer(strategy='mean')  # Standard: Mittelwert für Imputation
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    data_imputed = data.copy()
    data_imputed[numerical_cols] = imputer.fit_transform(data[numerical_cols])
    print("Fehlende Werte nach Imputation:")
    print(data_imputed.isnull().sum())

# Beispielaufruf für Naive Bayes
check_naive_bayes(data)

##############################
#%%# Check Neural Network ####
##############################

def check_neural_networks(data):
    # 1. Skalierung der numerischen Daten (für Neural Networks erforderlich)
    print("\nSkalierung der numerischen Daten (Standardabweichung vor und nach der Skalierung):")
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    numerical_data = data[numerical_cols]
    scaled_data = scaler.fit_transform(numerical_data)
    print("Standardabweichung vor Skalierung:", numerical_data.std())
    print("Standardabweichung nach Skalierung:", scaled_data.std(axis=0))

    # 2. Imputation fehlender Werte (für Neural Networks wichtig)
    print("\nImputation fehlender Werte (nur für numerische Spalten):")
    imputer = SimpleImputer(strategy='mean')  # Standard: Mittelwert für Imputation
    data_imputed = data.copy()
    data_imputed[numerical_cols] = imputer.fit_transform(data[numerical_cols])
    print("Fehlende Werte nach Imputation:")
    print(data_imputed.isnull().sum())

# Beispielaufruf für Neural Networks
check_neural_networks(data)

#############################
#%%# Check Random Forest ####
#############################

def check_random_forest(data):
    # 1. Überprüfen auf numerische und kategoriale Spalten
    print("Numerische Spalten (für Random Forest relevant):")
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    print(numerical_cols)

    print("\nKategoriale Spalten (für Random Forest relevant):")
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    print(categorical_cols)

    # 2. Überprüfen auf fehlende Werte
    print("\nFehlende Werte pro Spalte:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])

    # 3. Überprüfen auf die Verteilung der Zielspalte (nützlich für Klassifikation)
    if categorical_cols:
        target_col = categorical_cols[0]  # Annahme, dass die Zielspalte kategorial ist
        print("\nVerteilung der Zielspalte:")
        print(data[target_col].value_counts())

    # 4. Imputation fehlender Werte (Random Forest kann dies automatisch, aber hier zur Veranschaulichung)
    print("\nImputation fehlender Werte (nur für numerische Spalten):")
    imputer = SimpleImputer(strategy='mean')
    data_imputed = data.copy()
    data_imputed[numerical_cols] = imputer.fit_transform(data[numerical_cols])
    print("Fehlende Werte nach Imputation:")
    print(data_imputed.isnull().sum())

# Beispielaufruf für Random Forest
check_random_forest(data)
