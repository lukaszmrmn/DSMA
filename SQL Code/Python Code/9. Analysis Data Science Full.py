import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report, 
                             ConfusionMatrixDisplay, roc_auc_score, log_loss, 
                             precision_recall_curve, auc)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import plot_tree
from sklearn.inspection import PartialDependenceDisplay
import scikitplot as skplt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier


###################
#%%# Functions ####
###################


# Funktion zur Berechnung der Gini-Koeffizienten
def gini(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return 2 * auc - 1

def top_decile_lift(y_true, y_pred_proba):
    # Schritt 1: Berechne den Schwellenwert für das oberste Dezil (Top 10%)
    decile_threshold = np.percentile(y_pred_proba, 90)
    
    # Schritt 2: Finde die Indizes der Vorhersagen im Top Dezil
    top_decile_indices = y_pred_proba >= decile_threshold
    
    # Schritt 3: Berechne die Wahrscheinlichkeit der positiven Fälle im Top Dezil
    top_decile_positives = np.sum(y_true[top_decile_indices] == 1)
    total_in_decile = np.sum(top_decile_indices)
    prob_positive_in_decile = top_decile_positives / total_in_decile if total_in_decile > 0 else 0
    
    # Schritt 4: Berechne die Gesamtwahrscheinlichkeit der positiven Fälle
    total_positives = np.sum(y_true == 1)
    prob_positive_overall = total_positives / len(y_true) if len(y_true) > 0 else 0
    
    # Schritt 5: Berechne den Top Decile Lift (TDL)
    tdl = prob_positive_in_decile / prob_positive_overall if prob_positive_overall > 0 else 0
    return tdl



############################
#%%# Preparing the data ####
############################

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

data = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Working Data/Data_Ready_Tidy_2025-01-01_18-07-21.csv")
data['restaurants_price_range'] = data['restaurants_price_range'].astype('category')

X = data.drop('stars_above_average', axis = 1)
y = data['stars_above_average']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)




#################################################################
#%%# Preparing the data for Random Forest Feature Importance ####
#################################################################



data_1 = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Working Data/Combined_Tidy_Data_2024-12-30_14-22-52.csv")

# Spaltennamen aus data extrahieren
data_columns = data.columns.tolist()

# Nur die gemeinsamen Spalten in beiden DataFrames auswählen
common_columns = list(set(data_columns).intersection(data_1.columns))

# Werte aus data_1 übernehmen, wenn die Spalten in beiden DataFrames existieren
data[common_columns] = data_1[common_columns]



##########################
#%%## Random Forest ######
##########################

rfc = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=2, min_samples_leaf =4, random_state=42)
rfc.fit(X_train, y_train)
rf_pred_proba = rfc.predict_proba(X_test)

# Cumulative Gains Chart for Random Forrest
plot_rf = skplt.metrics.plot_cumulative_gain(y_test, rf_pred_proba)
ax = plt.gca()
lines = ax.get_lines()
for line in lines:
    if "Class 0" in line.get_label():
        line.remove()
plt.title("Cumulative Gains Chart - Random Forest Classifier")
plt.legend(["Random Forest Classifier", "Random Guess"], loc="lower right")
plt.show()

# Gini Coefficient for Random Forest
gini_rf = gini(y_test, rfc.predict(X_test))
print(f"Gini Coefficient - Random Forest: {gini_rf}")

# Print additional evaluation metrics
print("Train Accuracy ::", accuracy_score(y_train, rfc.predict(X_train)))
print("Test Accuracy  ::", accuracy_score(y_test, rfc.predict(X_test)))
print(classification_report(y_test, rfc.predict(X_test), target_names=['0', '1']))

# Confusion Matrix for Random Forest
print("Confusion Matrix - Random Forest:")
rf_cm = confusion_matrix(y_test, rfc.predict(X_test))
ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=['Below Average', 'Above Average']).plot(cmap='Blues', colorbar=False, values_format='d')
plt.title("Confusion Matrix - Random Forest")
plt.show()


##############################################
#%%## Random Forest Variable Importance ######
##############################################

# Wählen Sie einen Baum aus dem Random Forest (z.B. der erste Baum)
tree = rfc.estimators_[0]  # Index 0 für den ersten Baum, kannst auch einen anderen Baum wählen

# Plotten des Entscheidungsbaums
plt.figure(figsize=(200, 100))  # Größe des Plots
plot_tree(tree, filled=True, feature_names=X_train.columns, class_names=['Below Average', 'Above Average'], rounded=True, fontsize=12)
plt.title("Final Decision Tree - Random Forest")
plt.show()


#### Variable Importance

# Schriftgrößen anpassen
title_fontsize = 20        # Schriftgröße für den Titel
label_fontsize = 16        # Schriftgröße für Achsentitel
tick_fontsize = 14         # Schriftgröße für Tick-Labels

# Limit für die Anzahl der angezeigten Features
top_n_features = 10  # Anzahl der wichtigsten Features, die im Plot angezeigt werden sollen

# Extrahieren der Feature Importance
importances = rfc.feature_importances_
features = X.columns

# Sortieren und Anzeigen der Feature Importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Optional: Begrenzen auf die top_n_features
importance_df = importance_df.head(top_n_features)

# Ausgabe der Feature Importance
print(importance_df)

# Optional: Plotten der Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance', fontsize=label_fontsize)
plt.ylabel('Features', fontsize=label_fontsize)
plt.title('Feature Importance from Random Forest', fontsize=title_fontsize)
plt.gca().invert_yaxis()  # Wichtige Features oben
plt.tick_params(axis='both', labelsize=tick_fontsize)  # Schriftgröße für die Achsen

# Zeige das Plot an
plt.show()


##### Variable Importance Richtung

features_of_interest = [0, 1]  # Indizes der Features oder Namen
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(rfc, X, features_of_interest, ax=ax)
plt.suptitle("Partial Dependence Plots for Random Forest", fontsize=16)
plt.show()

features_of_interest = [3, 2]  # Indizes der Features oder Namen
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(rfc, X, features_of_interest, ax=ax)
plt.suptitle("Partial Dependence Plots for Random Forest", fontsize=16)
plt.show()

features_of_interest = [4, 6]  # Indizes der Features oder Namen
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(rfc, X, features_of_interest, ax=ax)
plt.suptitle("Partial Dependence Plots for Random Forest", fontsize=16)
plt.show()

features_of_interest = [5, 9]  # Indizes der Features oder Namen
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(rfc, X, features_of_interest, ax=ax)
plt.suptitle("Partial Dependence Plots for Random Forest", fontsize=16)
plt.show()

features_of_interest = [7, 45]  # Indizes der Features oder Namen
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(rfc, X, features_of_interest, ax=ax)
plt.suptitle("Partial Dependence Plots for Random Forest", fontsize=16)
plt.show()

#%%


# Merkmale, die geplottet werden sollen (jeweils Paare von Indizes)
feature_groups = [
    [0, 1],
    [3, 2],
    [4, 6],
    [5, 9],
    [7, 45]
]

# Beschreibungen der Features (Englisch)
feature_descriptions = [
    ["Food and Drinks Rating", "Service Rating"],
    ["Price Rating", "Atmosphere Rating"],
    ["Max. Temperature", "Daylight Duration"],
    ["Sunshine Duration", "Review Count"],
    ["Precipitation", "Alcohol Full Bar"]
]

# Anzahl der Zeilen
n_rows = len(feature_groups)

# Erstellen der Subplots mit gleicher Spaltenbreite
fig, axes = plt.subplots(
    n_rows, 2, figsize=(10, 16), 
    gridspec_kw={'wspace': 0.3, 'hspace': 0.4}  # Abstand zwischen den Plots
)

# Loop durch die Gruppen und Erstellen der Partial Dependence Plots
for i, (features, descriptions) in enumerate(zip(feature_groups, feature_descriptions)):
    # Achsen für die aktuelle Zeile
    ax1, ax2 = axes[i]

    # Plot für das erste Feature
    PartialDependenceDisplay.from_estimator(rfc, X, [features[0]], ax=ax1)
    ax1.set_title(f"PDP for {descriptions[0]}")
    
    # Plot für das zweite Feature
    PartialDependenceDisplay.from_estimator(rfc, X, [features[1]], ax=ax2)
    ax2.set_title(f"PDP for {descriptions[1]}")

# Gesamt-Titel für das Diagramm
fig.suptitle("Partial Dependence Plots for Random Forest", fontsize=16, y=0.95)

# Platzanpassung, um Achsen und Titel sauber anzuordnen
plt.tight_layout(rect=[0, 0, 1, 0.94])  # Platz für den Gesamttitel lassen
plt.show()



#############################################
#%%## Random Forest Optimal Parameters ######
#############################################

# Definiere die Parameterverteilung für Random Search
param_dist = {
    'max_depth': [5, 10, 15, 20, 25],  # Liste von möglichen max_depth-Werten
    'n_estimators': [100, 200, 300],        # Anzahl der Bäume
    'min_samples_split': [2, 5, 10],         # Minimum Anzahl Proben für den Split
    'min_samples_leaf': [1, 2, 4]            # Minimum Anzahl Proben für das Blatt
}

# Initialisiere das RandomForest-Modell
rfc = RandomForestClassifier(random_state=42)

# Initialisiere RandomizedSearchCV mit Kreuzvalidierung
random_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist, n_iter=100, cv=5, 
                                   n_jobs=-1, scoring='accuracy', random_state=42)

# Führe die Random-Suche aus
random_search.fit(X_train, y_train)

# Gebe das beste Ergebnis aus
print(f"Beste Parameter: {random_search.best_params_}")
print(f"Beste Genauigkeit: {random_search.best_score_}")

# Teste das Modell mit den besten Parametern
best_rfc = random_search.best_estimator_
test_accuracy = best_rfc.score(X_test, y_test)
print(f"Test-Genauigkeit mit besten Parametern: {test_accuracy}")



###########################
#%%## Neural Network ######
###########################

early_stopping = EarlyStopping(
    monitor='val_binary_accuracy', 
    patience=20,                  
    restore_best_weights=True
)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=2048, validation_split=0.2, verbose=0, callbacks=[early_stopping])
nn_pred_proba = model.predict(X_test)



# Cumulative Gains Chart für Neural Network
nn_pred_proba_2d = np.hstack([1 - nn_pred_proba, nn_pred_proba])
plot_nn = skplt.metrics.plot_cumulative_gain(y_test, nn_pred_proba_2d)
ax = plt.gca()
lines = ax.get_lines()
for line in lines:
    if "Class 0" in line.get_label():
        line.remove()
plt.title("Cumulative Gains Chart - Neural Network Classifier")
plt.legend(["Neural Network Classifier", "Random Guess"], loc="lower right")
plt.show()

# Gini Coefficient for Neural Network
gini_nn = gini(y_test, (nn_pred_proba > 0.5).astype(int))
print(f"Gini Coefficient - Neural Network: {gini_nn}")

# Print additional evaluation metrics
print("Train Accuracy ::", accuracy_score(y_train, (model.predict(X_train) > 0.5).astype(int)))
print("Test Accuracy  ::", accuracy_score(y_test, (nn_pred_proba > 0.5).astype(int)))
print(classification_report(y_test, (nn_pred_proba > 0.5).astype(int), target_names=['0', '1']))

# Confusion Matrix for Neural Network
print("Confusion Matrix - Neural Network:")
nn_cm = confusion_matrix(y_test, (nn_pred_proba > 0.5).astype(int))
ConfusionMatrixDisplay(confusion_matrix=nn_cm, display_labels=['Below Average', 'Above Average']).plot(cmap='Blues', colorbar=False, values_format='d')
plt.title("Confusion Matrix - Neural Network")
plt.show()

#################################################
#%%## Neural Network Hyperparameter-Tuning ######
#################################################

# Early Stopping definieren
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=10,                  
    restore_best_weights=True
)

# Definiere eine Funktion, um das Modell zu erstellen
def create_model(num_layers=1, num_neurons=[32]):
    model = Sequential()
    # Erste Schicht, mit der Eingabe
    model.add(Dense(num_neurons[0], activation='relu', input_dim=X_train.shape[1]))
    
    # Weitere Schichten hinzufügen, falls num_layers > 1
    for i in range(1, num_layers):
        model.add(Dense(num_neurons[i], activation='relu'))  # Hier variiert die Anzahl der Neuronen
    model.add(Dense(1, activation='sigmoid'))  # Ausgangsschicht
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrappen des Keras Modells für sklearn
model = KerasClassifier(build_fn=create_model, verbose=2)

param_grid = {
    'num_layers': [1, 2, 3],  # Anzahl der Layers
    'num_neurons': [
        # Für 1 Layer: Einfache Neuronenkombinationen
        [16], [32], [64], [128],  # Einzelne Neuronen pro Layer
        
        # Für 2 Layer: Kombinationen von Neuronen für aufsteigend und absteigend
        [16, 32], [32, 16],  # Aufsteigend und absteigend
        [32, 64], [64, 32],  # Aufsteigend und absteigend
        [64, 128], [128, 64],  # Aufsteigend und absteigend
        
        # Für 3 Layer: Kombinationen von Neuronen für aufsteigend und absteigend
        [16, 32, 64], [64, 32, 16],  # Aufsteigend und absteigend
        [32, 64, 128], [128, 64, 32],  # Aufsteigend und absteigend
        [16, 32, 128], [128, 32, 16],  # Aufsteigend und absteigend
        [32, 64, 128], [128, 64, 32],  # Aufsteigend und absteigend
    ],  # Kombiniere unterschiedliche Neuronenkombinationen pro Layer
    'epochs': [200],  # Nur 200 Epochen
    'batch_size': [512, 1024, 2048]  # Große Batch-Größen
}


# RandomizedSearchCV durchführen
randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=162, cv=3)
randomized_search.fit(X_train, y_train, validation_split=0.2, callbacks=[early_stopping])

# Beste Hyperparameter anzeigen
print(f"Beste Anzahl an Layern: {randomized_search.best_params_['num_layers']}")
print(f"Beste Anzahl an Neuronen: {randomized_search.best_params_['num_neurons']}")
print(f"Beste Batchgröße: {randomized_search.best_params_['batch_size']}")
print(f"Beste Anzahl an Epochen: {randomized_search.best_params_['epochs']}")

# Beste Accuracy auf dem Trainingsdatensatz ausgeben
best_model = randomized_search.best_estimator_
train_accuracy = best_model.score(X_train, y_train)
print(f"Beste Trainings-Accuracy: {train_accuracy:.4f}")


###########################################
#%%## K-Nearest Neighbors Classifier ######
###########################################

knn_classifier = KNeighborsClassifier(n_neighbors=56)
knn_classifier.fit(X_train, y_train)
knn_pred_proba = knn_classifier.predict_proba(X_test)

# Cumulative Gains Chart für KNN
plot_knn = skplt.metrics.plot_cumulative_gain(y_test, knn_pred_proba)
ax = plt.gca()
lines = ax.get_lines()
for line in lines:
    if "Class 0" in line.get_label():
        line.remove()
plt.title("Cumulative Gains Chart - KNN Classifier")
plt.legend(["KNN Classifier", "Random Guess"], loc="lower right")
plt.show()

# Gini Coefficient for KNN
gini_knn = gini(y_test, knn_classifier.predict(X_test))
print(f"Gini Coefficient - KNN: {gini_knn}")

# Print additional evaluation metrics
print("Train Accuracy ::", accuracy_score(y_train, knn_classifier.predict(X_train)))
print("Test Accuracy  ::", accuracy_score(y_test, knn_classifier.predict(X_test)))
print(classification_report(y_test, knn_classifier.predict(X_test), target_names=['0', '1']))

# Confusion Matrix for KNN
print("Confusion Matrix - KNN:")
knn_cm = confusion_matrix(y_test, knn_classifier.predict(X_test))
ConfusionMatrixDisplay(confusion_matrix=knn_cm, display_labels=['Below Average', 'Above Average']).plot(cmap='Blues', colorbar=False, values_format='d')
plt.title("Confusion Matrix - KNN")
plt.show()

###########################################################
#%%## K-Nearest Neighbors Classifier Optimal K-Value ######
###########################################################



# Bereich der k-Werte in 5er-Schritten
k_values = range(1, 100, 5)
cv_scores = []

# Cross-Validation für jeden k-Wert
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')  # 5-fold CV
    cv_scores.append(scores.mean())

# Optimalen k-Wert ermitteln
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")

# Plot der Cross-Validation Ergebnisse
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker='o', linestyle='-')
plt.xlabel('Number of Neighbors (k)', fontsize=12)
plt.ylabel('Cross-Validated Accuracy', fontsize=12)
plt.title('KNN Performance Across Different k Values', fontsize=14)
plt.xticks(k_values)
plt.grid(alpha=0.5)
plt.show()

# Trainiere KNN mit dem optimalen k-Wert
knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)
knn_classifier.fit(X_train, y_train)

# Teste den KNN-Classifier
knn_pred_proba = knn_classifier.predict_proba(X_test)
knn_test_accuracy = knn_classifier.score(X_test, y_test)
print(f"Test Accuracy with k={optimal_k}: {knn_test_accuracy:.4f}")

###################################
#%%## Naive Bayes Classifier ######
###################################

# Anzahl der besten Features (hier: 12)
k_best = 12

# Wähle die besten k Features aus
selector = SelectKBest(score_func=f_classif, k=k_best)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Oversample die Minority Class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

# Bestes var_smoothing aus der GridSearch
best_var_smoothing = 0.004094915062380427  # Dieser Wert wurde durch GridSearch gefunden

# Trainiere das Naive Bayes Modell mit dem besten var_smoothing-Wert
nb_classifier = GaussianNB(var_smoothing=best_var_smoothing)
nb_classifier.fit(X_train_resampled, y_train_resampled)

# Vorhersage der Wahrscheinlichkeiten auf den Testdaten
nb_pred_proba = nb_classifier.predict_proba(X_test_selected)

# Cumulative Gains Chart für Naive Bayes
plot_knn = skplt.metrics.plot_cumulative_gain(y_test, nb_pred_proba)
ax = plt.gca()
lines = ax.get_lines()
for line in lines:
    if "Class 0" in line.get_label():
        line.remove()
plt.title("Cumulative Gains Chart - Naive Bayes Classifier")
plt.legend(["Naive Bayes Classifier", "Random Guess"], loc="lower right")
plt.show()

# Gini Coefficient for Naive Bayes
gini_nb = gini(y_test, nb_classifier.predict(X_test_selected))
print(f"Gini Coefficient - Naive Bayes: {gini_nb}")

# Print additional evaluation metrics
print("Train Accuracy ::", accuracy_score(y_train, nb_classifier.predict(X_train_selected)))
print("Test Accuracy  ::", accuracy_score(y_test, nb_classifier.predict(X_test_selected)))
print(classification_report(y_test, nb_classifier.predict(X_test_selected), target_names=['0', '1']))

# Confusion Matrix for Naive Bayes
print("Confusion Matrix - Naive Bayes:")
nb_cm = confusion_matrix(y_test, nb_classifier.predict(X_test_selected))
ConfusionMatrixDisplay(confusion_matrix=nb_cm, display_labels=['Below Average', 'Above Average']).plot(cmap='Blues', colorbar=False, values_format='d')
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

#####################################################
#%%## Naive Bayes Classifier Feature Selection ######
#####################################################


# Anzahl der Features testen (z. B. die besten 5, 10, 15 ...)
num_features = range(5, X_train.shape[1] + 1)

# Liste zur Speicherung der Cross-Validation-Scores
cv_scores = []
feature_scores = []  # Liste zur Speicherung der Scores

for n in num_features:
    selector = SelectKBest(score_func=f_classif, k=n)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    nb_classifier = GaussianNB()
    scores = cross_val_score(nb_classifier, X_train_selected, y_train, cv=5, scoring='accuracy')
    
    cv_scores.append(scores.mean())

    # Speichern der Feature-Scores
    feature_scores.append(selector.scores_)

# Output results
optimal_features = num_features[np.argmax(cv_scores)]
best_accuracy = max(cv_scores)

print(f"Optimal number of features: {optimal_features}")
print(f"Best cross-validated accuracy: {best_accuracy:.4f}")

######################################################
#%%## Naive Bayes Classifier Feature Importance ######
######################################################

# Anzeige der Scores der Features (für die beste Anzahl von Features)
selector_best = SelectKBest(score_func=f_classif, k=optimal_features)
X_train_selected_best = selector_best.fit_transform(X_train, y_train)

# Zeige die Scores der ausgewählten Features an
feature_names = X_train.columns  # Wenn X_train ein DataFrame ist, um die Feature-Namen zu bekommen
sorted_features = sorted(zip(selector_best.scores_, feature_names), reverse=True)

print("\nTop Features by Score:")
for score, feature in sorted_features:
    print(f"Feature: {feature}, Score: {score:.4f}")

# Optional: Plotten der Top 12 Features
importance_df_nb = pd.DataFrame({
    'Feature': feature_names,
    'Importance': selector_best.scores_
})

# Sortiere nach Wichtigkeit (absteigend)
importance_df_nb = importance_df_nb.sort_values(by='Importance', ascending=False)

# Plotten der Top 12 Features
plt.figure(figsize=(10, 6))
plt.barh(importance_df_nb['Feature'][:12], importance_df_nb['Importance'][:12])  # Nur die Top 12 Features
plt.xlabel('Feature Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.title('Feature Importance from Naive Bayes', fontsize=16)
plt.gca().invert_yaxis()  # Wichtige Features oben
plt.tick_params(axis='both', labelsize=12)  # Schriftgröße für die Achsen
plt.show()

###############################################################
#%%## Naive Bayes Classifier Hyperparameter Optimization ######
###############################################################


# Define the parameter grid
param_grid = {
    'var_smoothing': np.logspace(-9, 0, 50)  # Test a range of values
}

# Initialize the model and GridSearchCV
nb_classifier = GaussianNB()
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameter and accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(f"Best var_smoothing: {best_params['var_smoothing']}")
print(f"Best cross-validated accuracy: {best_accuracy:.4f}")

# Test the best model
best_nb_classifier = grid_search.best_estimator_
nb_pred_proba = best_nb_classifier.predict_proba(X_test)

#####################################################################################
#%%## Naive Bayes Classifier Hyperparameter Optimization and Feature Selection ######
#####################################################################################

# Anzahl der besten Features (hier: 12)
k_best = 12

# Wähle die besten k Features aus
selector = SelectKBest(score_func=f_classif, k=k_best)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Cross-Validation für Feature-Auswahl mit k=12
cv_scores = cross_val_score(GaussianNB(), X_train_selected, y_train, cv=5, scoring='accuracy')
print(f"Best cross-validated accuracy with k=12 features: {cv_scores.mean():.4f}")

# Define the parameter grid for GridSearchCV
param_grid = {
    'var_smoothing': np.logspace(-9, 0, 50)  # Test a range of values for var_smoothing
}

# Initialize the model and GridSearchCV
nb_classifier = GaussianNB()
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model using only the selected features
grid_search.fit(X_train_selected, y_train)

# Best parameter and accuracy after Grid Search
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(f"Best var_smoothing: {best_params['var_smoothing']}")
print(f"Best cross-validated accuracy after Grid Search: {best_accuracy:.4f}")

# Test the best model on the test set
best_nb_classifier = grid_search.best_estimator_
nb_pred_proba = best_nb_classifier.predict_proba(X_test_selected)

##################################################################################################################
#%%## Naive Bayes Classifier Hyperparameter Optimization and Feature Selection and Handling Class Imbalance ######
##################################################################################################################

# Anzahl der besten Features (hier: 12)
k_best = 12

# Wähle die besten k Features aus
selector = SelectKBest(score_func=f_classif, k=k_best)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Oversample the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

# Bestes var_smoothing aus der GridSearch
best_var_smoothing = 0.004094915062380427  # Dieser Wert wurde durch GridSearch gefunden

# Trainiere das Naive Bayes Modell mit dem besten var_smoothing-Wert
nb_classifier = GaussianNB(var_smoothing=best_var_smoothing)
nb_classifier.fit(X_train_resampled, y_train_resampled)

# Vorhersage der Wahrscheinlichkeiten auf den Testdaten
nb_pred_proba = nb_classifier.predict_proba(X_test_selected)

# Vorhersage der Klassenlabels auf den Testdaten
nb_pred = nb_classifier.predict(X_test_selected)

# Berechne und gebe die Accuracy aus
accuracy = accuracy_score(y_test, nb_pred)
print(f"Accuracy on test data: {accuracy:.4f}")



####################################
#%%## Cumulative Gains Curves ######
####################################

# Schriftgrößen anpassen
title_fontsize = 20        # Schriftgröße für den Titel
label_fontsize = 16        # Schriftgröße für die Achsentitel
tick_fontsize = 14         # Schriftgröße für die Tick-Labels
legend_fontsize = 14       # Schriftgröße für die Legende

# Erstelle eine neue Abbildung
plt.figure(figsize=(10, 6))

# Plotte die Cumulative Gain Kurven
skplt.metrics.plot_cumulative_gain(y_test, rf_pred_proba, ax=plt.gca())
skplt.metrics.plot_cumulative_gain(y_test, nn_pred_proba_2d, ax=plt.gca())
skplt.metrics.plot_cumulative_gain(y_test, knn_pred_proba, ax=plt.gca())
skplt.metrics.plot_cumulative_gain(y_test, nb_pred_proba, ax=plt.gca())

# Berechne die AUC für jede Kurve
rf_x, rf_y = skplt.metrics.cumulative_gain_curve(y_test, rf_pred_proba[:, 1])
nn_x, nn_y = skplt.metrics.cumulative_gain_curve(y_test, nn_pred_proba_2d[:, 1])
knn_x, knn_y = skplt.metrics.cumulative_gain_curve(y_test, knn_pred_proba[:, 1])
nb_x, nb_y = skplt.metrics.cumulative_gain_curve(y_test, nb_pred_proba[:, 1])

rf_auc_cg = auc(rf_x, rf_y)
nn_auc_cg = auc(nn_x, nn_y)
knn_auc_cg = auc(knn_x, knn_y)
nb_auc_cg = auc(nb_x, nb_y)

# Entferne alle Linien, die "Class 0" beinhalten
ax = plt.gca()
lines = ax.get_lines()
for line in lines:
    if "Class 0" in line.get_label():
        line.remove()

# Linien anpassen
linewidth = 2
lines[1].set_color('blue')  # Random Forest
lines[1].set_linewidth(linewidth)

lines[4].set_color('green')  # Neural Network
lines[4].set_linewidth(linewidth)

lines[7].set_color('red')  # KNN
lines[7].set_linewidth(linewidth)

lines[10].set_color('purple')  # Naive Bayes
lines[10].set_linewidth(linewidth)

# Legendenhandles und Labels mit AUC-Werten
handles = [lines[i] for i in [1, 4, 7, 10]]
labels = [
    f"Random Forest (AUC = {rf_auc_cg:.2f})",
    f"Neural Network (AUC = {nn_auc_cg:.2f})",
    f"K-Nearest Neighbors (AUC = {knn_auc_cg:.2f})",
    f"Naive Bayes (AUC = {nb_auc_cg:.2f})"
]

# Titel, Labels und Legende anpassen
plt.title("Cumulative Gains Curves for Various Models", fontsize=title_fontsize)
plt.xlabel("Percentage of Samples", fontsize=label_fontsize)
plt.ylabel("Percentage of Positive Target", fontsize=label_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.legend(handles=handles, labels=labels, loc="lower right", fontsize=legend_fontsize)

# Zeige den Plot an
plt.show()


#############################
#%%## Confusion Matrix ######
#############################

# Schriftgrößen anpassen
title_fontsize = 22  # Schriftgröße für Titel
label_fontsize = 22  # Schriftgröße für Achsentitel und Labels
tick_fontsize = 16   # Schriftgröße für Tick-Labels
text_fontsize = 20   # Schriftgröße für die Zahlen in der Confusion Matrix

# Plot all Confusion Matrices in one figure
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Random Forest
disp_rf = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=['Below Average', 'Above Average'])
disp_rf.plot(ax=axs[0, 0], cmap='Blues', colorbar=False, values_format='d')
axs[0, 0].set_title("Confusion Matrix - Random Forest", fontsize=title_fontsize)
axs[0, 0].set_xlabel('Predicted Label', fontsize=label_fontsize)
axs[0, 0].set_ylabel('True Label', fontsize=label_fontsize)
axs[0, 0].tick_params(axis='both', labelsize=tick_fontsize)
for text in disp_rf.text_.ravel():
    text.set_fontsize(text_fontsize)

# Neural Network
disp_nn = ConfusionMatrixDisplay(confusion_matrix=nn_cm, display_labels=['Below Average', 'Above Average'])
disp_nn.plot(ax=axs[0, 1], cmap='Blues', colorbar=False, values_format='d')
axs[0, 1].set_title("Confusion Matrix - Neural Network", fontsize=title_fontsize)
axs[0, 1].set_xlabel('Predicted Label', fontsize=label_fontsize)
axs[0, 1].set_ylabel('True Label', fontsize=label_fontsize)
axs[0, 1].tick_params(axis='both', labelsize=tick_fontsize)
for text in disp_nn.text_.ravel():
    text.set_fontsize(text_fontsize)

# K-Nearest Neighbors
disp_knn = ConfusionMatrixDisplay(confusion_matrix=knn_cm, display_labels=['Below Average', 'Above Average'])
disp_knn.plot(ax=axs[1, 0], cmap='Blues', colorbar=False, values_format='d')
axs[1, 0].set_title("Confusion Matrix - KNN", fontsize=title_fontsize)
axs[1, 0].set_xlabel('Predicted Label', fontsize=label_fontsize)
axs[1, 0].set_ylabel('True Label', fontsize=label_fontsize)
axs[1, 0].tick_params(axis='both', labelsize=tick_fontsize)
for text in disp_knn.text_.ravel():
    text.set_fontsize(text_fontsize)

# Naive Bayes
disp_nb = ConfusionMatrixDisplay(confusion_matrix=nb_cm, display_labels=['Below Average', 'Above Average'])
disp_nb.plot(ax=axs[1, 1], cmap='Blues', colorbar=False, values_format='d')
axs[1, 1].set_title("Confusion Matrix - Naive Bayes", fontsize=title_fontsize)
axs[1, 1].set_xlabel('Predicted Label', fontsize=label_fontsize)
axs[1, 1].set_ylabel('True Label', fontsize=label_fontsize)
axs[1, 1].tick_params(axis='both', labelsize=tick_fontsize)
for text in disp_nb.text_.ravel():
    text.set_fontsize(text_fontsize)

plt.tight_layout()
plt.show()




##########################
#%%## Deviance Loss ######
##########################

# Für Random Forest
rf_pred_proba_class1 = rf_pred_proba[:, 1]  # Vorhersagewahrscheinlichkeiten für Klasse 1
rf_dl = log_loss(y_test, rf_pred_proba)

# Für Neural Network
nn_pred_proba_class1 = nn_pred_proba_2d[:, 1]  # Vorhersagewahrscheinlichkeiten für Klasse 1
nn_dl = log_loss(y_test, nn_pred_proba_2d)

# Für KNN
knn_pred_proba_class1 = knn_pred_proba[:, 1]  # Vorhersagewahrscheinlichkeiten für Klasse 1
knn_dl = log_loss(y_test, knn_pred_proba)

# Für Naive Bayes
nb_pred_proba_class1 = nb_pred_proba[:, 1]  # Vorhersagewahrscheinlichkeiten für Klasse 1
nb_dl = log_loss(y_test, nb_pred_proba)

# Ausgabe der TDL-Werte
print(f"DL (Deviance Loss) for Random Forest: {rf_dl}")
print(f"DL (Deviance Loss) for Neural Network: {nn_dl}")
print(f"DL (Deviance Loss) for KNN: {knn_dl}")
print(f"DL (Deviance Loss) for Naive Bayes: {nb_dl}")


###########################
#%%## Top Decil Lift ######
###########################

# Random Forest
rf_tdl = top_decile_lift(y_test, rf_pred_proba[:, 1])  # Wir nehmen die Wahrscheinlichkeiten der positiven Klasse (Klasse 1)
print(f"Top Decile Lift for Random Forest: {rf_tdl}")

# Neural Network
nn_tdl = top_decile_lift(y_test, nn_pred_proba_2d[:, 1])
print(f"Top Decile Lift for Neural Network: {nn_tdl}")

# KNN
knn_tdl = top_decile_lift(y_test, knn_pred_proba[:, 1])
print(f"Top Decile Lift for KNN: {knn_tdl}")

# Naive Bayes
nb_tdl = top_decile_lift(y_test, nb_pred_proba[:, 1])
print(f"Top Decile Lift for Naive Bayes: {nb_tdl}")


#######################
#%%## ROC Curves ######
#######################

# Schriftgrößen anpassen
title_fontsize = 20        # Schriftgröße für den Titel
label_fontsize = 16        # Schriftgröße für die Achsentitel
tick_fontsize = 14         # Schriftgröße für die Tick-Labels
legend_fontsize = 14       # Schriftgröße für die Legende

# Erstelle eine neue Abbildung
plt.figure(figsize=(10, 6))

# Plotte die ROC-Kurven
skplt.metrics.plot_roc(y_test, rf_pred_proba, ax=plt.gca())
skplt.metrics.plot_roc(y_test, nn_pred_proba_2d, ax=plt.gca())
skplt.metrics.plot_roc(y_test, knn_pred_proba, ax=plt.gca())
skplt.metrics.plot_roc(y_test, nb_pred_proba, ax=plt.gca())

# Entferne alle Linien, die "class 0", "micro-average" oder "macro-average" beinhalten
ax = plt.gca()
lines = ax.get_lines()

for line in lines:
    if "class 0" in line.get_label():
        line.remove()
    if "micro-average" in line.get_label():
        line.remove()
    if "macro-average" in line.get_label():
        line.remove()

# Berechne den AUC-Wert für jedes Modell
rf_auc = round(roc_auc_score(y_test, rf_pred_proba[:, 1]), 2)  # Random Forest
nn_auc = round(roc_auc_score(y_test, nn_pred_proba), 2)  # Neural Network
knn_auc = round(roc_auc_score(y_test, knn_pred_proba[:, 1]), 2)  # KNN
nb_auc = round(roc_auc_score(y_test, nb_pred_proba[:, 1]), 2)  # Naive Bayes

# Linien anpassen
linewidth = 2

lines[1].set_color('blue')  # Random Forest
lines[1].set_linewidth(linewidth)  # Linienstärke für Random Forest
lines[1].set_label(f"Random Forest (AUC = {rf_auc})")

lines[6].set_color('green')  # Neural Network
lines[6].set_linewidth(linewidth)  # Linienstärke für Neural Network
lines[6].set_label(f"Neural Network (AUC = {nn_auc})")

lines[11].set_color('red')  # KNN
lines[11].set_linewidth(linewidth)  # Linienstärke für KNN
lines[11].set_label(f"KNN (AUC = {knn_auc})")

lines[16].set_color('purple')  # Naive Bayes
lines[16].set_linewidth(linewidth)  # Linienstärke für Naive Bayes
lines[16].set_label(f"Naive Bayes (AUC = {nb_auc})")

# Legendenhandles und Labels
indices = [1, 6, 11, 16]
handles = []
labels = []

for i in indices:
    line = lines[i]
    handles.append(line)   # Füge die Linie zum Handle hinzu
    labels.append(line.get_label())  # Füge das Label der Linie hinzu

# Titel, Labels und Legende anpassen
plt.title("ROC Curves for Various Models", fontsize=title_fontsize)
plt.xlabel("False Positive Rate", fontsize=label_fontsize)
plt.ylabel("True Positive Rate", fontsize=label_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.legend(handles=handles, labels=labels, loc="lower right", fontsize=legend_fontsize)

# Zeige den Plot an
plt.show()





########################
#%%## Lift Curves ######
########################

# Schriftgrößen anpassen
title_fontsize = 20        # Schriftgröße für den Titel
label_fontsize = 16        # Schriftgröße für die Achsentitel
tick_fontsize = 14         # Schriftgröße für die Tick-Labels
legend_fontsize = 14       # Schriftgröße für die Legende

# Erstelle eine neue Abbildung
plt.figure(figsize=(10, 6))

# Plotte die Lift-Kurven
skplt.metrics.plot_lift_curve(y_test, rf_pred_proba, ax=plt.gca())
skplt.metrics.plot_lift_curve(y_test, nn_pred_proba_2d, ax=plt.gca())
skplt.metrics.plot_lift_curve(y_test, knn_pred_proba, ax=plt.gca())
skplt.metrics.plot_lift_curve(y_test, nb_pred_proba, ax=plt.gca())

#for i, line in enumerate(lines):
#    print(f"Linie {i}:")
#    print(f"  Label: {line.get_label()}")
#    print(f"  Farbe: {line.get_color()}")
#    print(f"  Linienstil: {line.get_linestyle()}")
#    print(f"  Datenpunkte: {line.get_data()}")
#    print("-" * 30)

# Entferne alle Linien, die "Class 0" beinhalten
ax = plt.gca()
lines = ax.get_lines()
for line in lines:
    if "Class 0" in line.get_label():
        line.remove()

# Linien anpassen
linewidth = 2

lines[1].set_color('blue')  # Random Forest
lines[1].set_linewidth(linewidth)  # Linienstärke für Random Forest

lines[4].set_color('green')  # Neural Network
lines[4].set_linewidth(linewidth)  # Linienstärke für Neural Network

lines[7].set_color('red')  # KNN
lines[7].set_linewidth(linewidth)  # Linienstärke für KNN

lines[10].set_color('purple')  # Naive Bayes
lines[10].set_linewidth(linewidth)  # Linienstärke für Naive Bayes

# Überprüfen und Auswählen der Linien mit Indizes 1, 4, 7 und 10
indices = [1, 4, 7, 10]
handles = []
labels = ["Random Forest", "Neural Network", "K-Nearest Neighbors", "Naive Bayes"]

for i in indices:
    line = lines[i]
    handles.append(line)   # Füge die Linie zum Handle hinzu
    labels.append(line.get_label())  # Füge das Label der Linie hinzu

# Titel, Labels und Legende anpassen
plt.title("Lift Curves for Various Models", fontsize=title_fontsize)
plt.xlabel("Percentage of Samples", fontsize=label_fontsize)
plt.ylabel("Lift", fontsize=label_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.legend(handles=handles, labels=labels, loc="lower right", fontsize=legend_fontsize)

# Zeige den Plot an
plt.show()




####################################
#%%## Precision-Recall Curves ######
####################################

# Schriftgrößen anpassen
title_fontsize = 20        # Schriftgröße für den Titel
label_fontsize = 16        # Schriftgröße für die Achsentitel
tick_fontsize = 14         # Schriftgröße für die Tick-Labels
legend_fontsize = 14       # Schriftgröße für die Legende

# Erstelle eine neue Abbildung
plt.figure(figsize=(10, 6))

# Plotte die Precision-Recall Kurven
skplt.metrics.plot_precision_recall_curve(y_test, rf_pred_proba, ax=plt.gca())
skplt.metrics.plot_precision_recall_curve(y_test, nn_pred_proba_2d, ax=plt.gca())
skplt.metrics.plot_precision_recall_curve(y_test, knn_pred_proba, ax=plt.gca())
skplt.metrics.plot_precision_recall_curve(y_test, nb_pred_proba, ax=plt.gca())

# Berechnung der Precision-Recall-Kurve und AUC für jedes Modell
precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_pred_proba[:, 1])
precision_nn, recall_nn, _ = precision_recall_curve(y_test, nn_pred_proba)
precision_knn, recall_knn, _ = precision_recall_curve(y_test, knn_pred_proba[:, 1])
precision_nb, recall_nb, _ = precision_recall_curve(y_test, nb_pred_proba[:, 1])

# Berechnung der AUC (x = Recall, y = Precision)
rf_auc_pr = auc(recall_rf, precision_rf)
nn_auc_pr = auc(recall_nn, precision_nn)
knn_auc_pr = auc(recall_knn, precision_knn)
nb_auc_pr = auc(recall_nb, precision_nb)

# Entferne alle Linien, die "class 0" oder andere irrelevante Labels haben
ax = plt.gca()
lines = ax.get_lines()
for line in lines:
    if "class 0" in line.get_label() or "micro-average" in line.get_label():
        line.remove()


#for i, line in enumerate(lines):
#    print(f"Linie {i}:")
#    print(f"  Label: {line.get_label()}")
#    print(f"  Farbe: {line.get_color()}")
#    print(f"  Linienstil: {line.get_linestyle()}")
#    print(f"  Datenpunkte: {line.get_data()}")
#    print("-" * 30)

# Linien anpassen
linewidth = 2
lines[1].set_color('blue')  # Random Forest
lines[1].set_linewidth(linewidth)

lines[4].set_color('green')  # Neural Network
lines[4].set_linewidth(linewidth)

lines[7].set_color('red')  # KNN
lines[7].set_linewidth(linewidth)

lines[10].set_color('purple')  # Naive Bayes
lines[10].set_linewidth(linewidth)

# Legendenhandles und Labels
handles = [lines[i] for i in [1, 4, 7, 10]]
labels = [
    f"Random Forest (AUC = {rf_auc_pr:.2f})",
    f"Neural Network (AUC = {nn_auc_pr:.2f})",
    f"K-Nearest Neighbors (AUC = {knn_auc_pr:.2f})",
    f"Naive Bayes (AUC = {nb_auc_pr:.2f})"
]

# Titel, Labels und Legende anpassen
plt.title("Precision-Recall Curves for Various Models", fontsize=title_fontsize)
plt.xlabel("Recall", fontsize=label_fontsize)
plt.ylabel("Precision", fontsize=label_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.legend(handles=handles, labels=labels, loc="lower right", fontsize=legend_fontsize)

# Zeige den Plot an
plt.show()


