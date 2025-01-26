import requests
import pandas as pd
import time
from collections import Counter
from time import sleep
import pickle  # Zum Speichern des Fortschritts
import os  # Zum Überprüfen, ob die Datei existiert


# Dateipfade
temp_file_path = "C:/Users/lukas/Desktop/DSMA/Working Data/Review_Business_Weather_Data_Temp.csv"
default_file_path = "C:/Users/lukas/Desktop/DSMA/Working Data/Review_Business_Data.csv"

# Datei auswählen basierend auf der Existenz von temp_file_path
if os.path.exists(temp_file_path):
    data = pd.read_csv(temp_file_path)
    print(f"Datei '{temp_file_path}' wurde geladen.")
else:
    data = pd.read_csv(default_file_path)
    print(f"Datei '{default_file_path}' wurde geladen.")
    
#data = data.head(50)  # Nur die ersten 50 Zeilen nehmen

#%%

# API-Konfiguration für die archive API
api_url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": None,
    "longitude": None,
    "start_date": "",  # Startdatum im Format: 2005-03-10
    "end_date": "",    # Enddatum im Format: 2005-03-10
    "hourly": "temperature_2m,weathercode",  # Temperatur und Wettercode zur Stunde
    "daily": "sunshine_duration,daylight_duration,rain_sum,snowfall_sum",  
    "timezone": "auto"             # Zeitzoneneinstellung
}

# Funktion zur Abfrage und Verarbeitung der Daten
def fetch_weather_data(date, latitude, longitude):
    formatted_date = date.split()[0]  # Nur das Datum extrahieren
    params['start_date'] = formatted_date
    params['end_date'] = formatted_date
    params['latitude'] = latitude
    params['longitude'] = longitude
    
    for attempt in range(retries):
        try:
            response = requests.get(api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                hourly_data = data['hourly']
                weather_codes = hourly_data['weathercode']
                most_common_code, _ = Counter(weather_codes).most_common(1)[0]
                weather_description = weather_code_to_description(most_common_code)
                max_temp = max(hourly_data['temperature_2m'])
                min_temp = min(hourly_data['temperature_2m'])
                
                # UPDATED CODE: Extrahiere tägliche Variablen
                daily_data = data['daily']
                sunshine_duration = daily_data['sunshine_duration'][0] / 3600 # In Minuten
                daylight_duration = daily_data['daylight_duration'][0] / 3600  # Umrechnen in Stunden
                rain_sum = daily_data['rain_sum'][0]  # Niederschlag in mm
                snowfall_sum = daily_data['snowfall_sum'][0]  # Schneefall in mm

                return max_temp, min_temp, weather_description, sunshine_duration, daylight_duration, rain_sum, snowfall_sum  # UPDATED CODE: Rückgabe erweitert
            else:
                print(f"API-Fehler {response.status_code}. Versuche erneut...")
                time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            print(f"Fehler: {e}. Warte {wait_time} Sekunden und versuche erneut...")
            time.sleep(wait_time)
    print("Maximale Anzahl von Versuchen erreicht. Weiter zur nächsten Zeile.")
    return None, None, None, None, None, None, None  

# Funktion zur Umwandlung des weather codes in eine Beschreibung
def weather_code_to_description(code):
    weather_descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Heavy rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return weather_descriptions.get(code, "Unknown weather")

# Fortschritt speichern
def save_progress(index):
    with open("progress.pkl", "wb") as f:
        pickle.dump(index, f)

# Fortschritt laden
def load_progress():
    try:
        with open("progress.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return 0

# Wetterdaten direkt dem DataFrame hinzufügen
start_time = time.time()  # Startzeit messen
total_rows = len(data)  # Gesamtanzahl der Zeilen im DataFrame
start_index = load_progress() - 10 # Fortschritt laden
retries = 80
wait_time = 15

for index, row in data.iloc[start_index:].iterrows():
    max_temp, min_temp, weather_description, sunshine_duration, daylight_duration, rain_sum, snowfall_sum = fetch_weather_data(  # UPDATED CODE: Zusätzliche Rückgabewerte
        row['date'], row['latitude'], row['longitude']
    )
    if max_temp is not None and min_temp is not None and weather_description is not None and sunshine_duration is not None:  # UPDATED CODE: Zusätzliche Bedingung
        data.at[index, 'max_temperature'] = max_temp
        data.at[index, 'min_temperature'] = min_temp
        data.at[index, 'weather_description'] = weather_description
        data.at[index, 'sunshine_duration'] = sunshine_duration  
        data.at[index, 'daylight_duration'] = daylight_duration  
        data.at[index, 'rain_sum'] = rain_sum  
        data.at[index, 'snowfall_sum'] = snowfall_sum 
        save_progress(index)
    else:
        print(f"Fehler bei Index {index}. Speichern und Stoppen")
        # Speichern des aktuellen Fortschritts
        data.to_csv("C:/Users/lukas/Desktop/DSMA/Working Data/Review_Business_Weather_Data_Temp.csv", index=False)
        save_progress(index - 1)
        print("Erfolgreich gespeichert und gestoppt")
        break
    
    # Timer und Fortschrittanzeige mit verbleibender Zeit in Minuten, falls länger als 600 Sekunden
    elapsed_time = time.time() - start_time
    completed_percentage = (index + 1) / total_rows * 100
    remaining_estimate = (elapsed_time / (index - start_index + 1)) * (total_rows - index)
    
    if remaining_estimate > 600:  # Wenn die geschätzte verbleibende Zeit über 600 Sekunden ist
        print(f"Progress: {index + 1}/{total_rows} | Elapsed Time: {elapsed_time:.2f}s | Completed: {completed_percentage:.2f}% | Estimated Remaining: {remaining_estimate / 60:.2f} minutes")
    else:
        print(f"Progress: {index + 1}/{total_rows} | Elapsed Time: {elapsed_time:.2f}s | Completed: {completed_percentage:.2f}% | Estimated Remaining: {remaining_estimate:.2f}s")
    
    # Wartezeit von 0.5 Sekunden
    #sleep(0.5)

# Gesamtdauer ausgeben
total_elapsed_time = time.time() - start_time
print(f"Gesamtdauer: {total_elapsed_time:.2f} Sekunden")



# Ergebnis speichern
data.to_csv("C:/Users/lukas/Desktop/DSMA/Working Data/Review_Business_Weather_Data.csv", index=False)

print("Wetterdaten erfolgreich hinzugefügt und gespeichert.")



