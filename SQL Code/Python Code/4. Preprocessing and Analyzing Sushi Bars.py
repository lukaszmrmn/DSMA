import pandas as pd
import numpy as np
import ast
from datetime import datetime


data = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Working Data/Business Data Sushi Bars.csv")

#data = data.head(10)


def analyze_column(column_name):
    print(f"Analyse der Spalte: {column_name}\n")
    # Einzigartige Werte
    unique_values = data[column_name].unique()
    print("Eindeutige Werte:")
    print(unique_values, "\n")
    
    # Häufigkeiten der Werte
    value_counts = data[column_name].value_counts()
    print("Häufigkeiten der Werte:")
    print(value_counts, "\n")
    
    # Anzahl der fehlenden Werte
    missing_count = data[column_name].isna().sum()
    print(f"Anzahl fehlender Werte: {missing_count}\n")


#%%

# Trennung der Kategorien in einzelne Wörter und Ausgabe der einzigartigen Wörter
unique_words = set()

# Spalte durchgehen und Wörter extrahieren
for categories in data['categories']:
    words = categories.split(',')  # Trennen der Wörter basierend auf Komma
    unique_words.update(word.strip().lower() for word in words)  # Wörter in Kleinbuchstaben und Duplikate entfernen

# Alle einzigartigen Wörter ausgeben
print("Unique words in 'categories':")
print(sorted(unique_words))  # Sortierte Liste der einzigartigen Wörter




#%%

# Klassische Sushi-Kategorien definieren
sushi_keywords = {
    'acai bowls', 'asian fusion', 'sushi bars', 'chinese', 'conveyor belt sushi', 'izakaya', 'japanese', 'japanese curry',
    'ramen', 'szechuan', 'shanghainese', 'teppanyaki', 'seafood', 'sushi',
    'soup', 'noodles', 'korean', 'live/raw food', 'poke', "thai", "chinese", "buffets", "hot pot"
}

# Funktion, um die Klassifizierung basierend auf Sushi-Kategorien durchzuführen
def classify_sushi_specialist_type(categories):
    if pd.isna(categories):
        return False
    categories_list = categories.lower().split(', ')
    sushi_count = sum(1 for keyword in sushi_keywords if keyword.strip() in categories_list)
    
    if sushi_count / len(categories_list) >= 0.5:
        return True
    else:
        return False

# Anwendung der Funktion auf die 'categories' Spalte
data['sushi_specialist'] = data['categories'].apply(classify_sushi_specialist_type)


#%%

analyze_column('categories')

analyze_column('sushi_specialist')

#%%

analyze_column('wifi')

#%%

# Umwandlung der Werte in der 'wifi' Spalte
def transform_wifi(value):
    if pd.isna(value):
        return np.nan  # Fehlende Werte bleiben NaN
    value = str(value).lower()  # Alle Werte klein schreiben für Konsistenz
    if 'free' in value:
        return True  # WiFi ist kostenlos
    elif 'no' in value or 'paid' in value:
        return False  # Kein WiFi oder WiFi ist kostenpflichtig

# Anwenden der Funktion auf die 'wifi' Spalte
data['wifi'] = data['wifi'].apply(transform_wifi)

data.rename(columns={'wifi': 'free_wifi'}, inplace=True)


#%%

analyze_column('free_wifi')


#%%

analyze_column('has_tv')

#%%

analyze_column('caters')

#%%

analyze_column('alcohol')

#%%

def clean_alcohol(cell):
    if pd.isna(cell):  # Überprüfe, ob der Wert NaN ist
        return None
    cell = str(cell).strip().lower()  # In Kleinbuchstaben und ohne Leerzeichen
    if "full_bar" in cell:
        return "full_bar"
    elif "beer_and_wine" in cell:
        return "beer_and_wine"
    elif "none" in cell:
        return "none"
    else:
        return None  # Für unbekannte Werte

# Wende die Funktion auf die Spalte an
data['alcohol'] = data['alcohol'].apply(clean_alcohol)

#%%

analyze_column('alcohol')

#%%

analyze_column('noise_level')

#%%

# Funktion zur Zusammenfassung der Kategorien
def clean_noise_level(value):
    # Entferne Präfixe wie "u'"
    value = str(value).replace("u'", "").replace("'", "").strip()
    
    # Gruppiere die Werte
    if value in ['quiet']:
        return 'quiet'
    elif value in ['average']:
        return 'average'
    elif value in ['loud']:
        return 'loud'
    elif value in ['very_loud']:
        return 'very_loud'
    else:
        return None  # Für unbekannte oder leere Werte

# Anwenden der Funktion auf die Spalte noise_level
data['noise_level'] = data['noise_level'].apply(clean_noise_level)

#%%

analyze_column('noise_level')

#%%

analyze_column('bike_parking')

#%%

analyze_column('good_for_kids')


#%%

analyze_column('outdoor_seating')


#%%

analyze_column('restaurants_attire')

#%%

# Nur 'casual' oder 'dressy' behalten
data['restaurants_attire'] = data['restaurants_attire'].replace({r"u'casual'": 'casual', r"'casual'": 'casual', r"u'dressy'": 'dressy', r"'dressy'": 'dressy'})
analyze_column('restaurants_attire')

#%%

analyze_column('restaurants_take_out')

#%%

analyze_column('restaurants_delivery')

#%%

analyze_column('restaurants_price_range')

#%%

analyze_column('restaurants_reservations')


#%%

analyze_column('restaurants_good_for_groups')

#%%

analyze_column('business_accepts_credit_cards')

#%%

analyze_column('hours_monday')

#%%

# Funktion zum Splitten der Stunden und Umwandeln in Zeitformat
def split_hours(hours):
    if pd.isna(hours):  # Falls keine Öffnungszeiten angegeben sind
        return pd.Series([None, None])
    
    try:
        open_time, close_time = hours.split('-')
        # Umwandlung der Stunden in das Zeitformat 'HH:MM'
        open_time = pd.to_datetime(open_time, format='%H:%M').time()
        close_time = pd.to_datetime(close_time, format='%H:%M').time()
        return pd.Series([open_time, close_time])
    except ValueError:
        return pd.Series([None, None])

# Für jede Stunden-Spalte die Funktion anwenden und die neuen Spalten erstellen
days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

for day in days_of_week:
    open_col = f'open_{day}'
    close_col = f'close_{day}'
    
    data[[open_col, close_col]] = data[f'hours_{day}'].apply(split_hours)


#%%

data.drop(columns=[f'hours_{day}' for day in days_of_week], inplace=True)


#%%

data['ambience'] = data['ambience'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
for key in data['ambience'].dropna().iloc[0].keys():
    data[f'ambience_{key}'] = data['ambience'].apply(lambda x: x.get(key) if isinstance(x, dict) else None)

#%%

data['good_for_meal'] = data['good_for_meal'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
for key in data['good_for_meal'].dropna().iloc[0].keys():
    data[f'good_for_meal_{key}'] = data['good_for_meal'].apply(lambda x: x.get(key) if isinstance(x, dict) else None)

#%%

data['business_parking'] = data['business_parking'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
for key in data['business_parking'].dropna().iloc[0].keys():
    data[f'business_parking_{key}'] = data['business_parking'].apply(lambda x: x.get(key) if isinstance(x, dict) else None)

#%%

data = data.drop(columns=["business_parking",'good_for_meal',"ambience"])

#%%

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data.to_csv(f"C:/Users/lukas/Desktop/DSMA/Working Data/Business_Data_Sushi_Bars_Clean_{timestamp}.csv", index=False)

