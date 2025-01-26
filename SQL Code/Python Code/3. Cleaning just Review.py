import pandas as pd
from datetime import datetime


data = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Working Data/Review_Business_Weather_Data.csv")


#%%

# if merged before remove business columns otherwise just the unnecesary Review columns

columns_to_drop = [
    "categories", "city", "state", "postal_code", "latitude", "longitude", "stars_y", 
    "review_count", "is_open", "byob", "wifi", "has_tv", "music", "caters", "alcohol", 
    "smoking", "ambience", "coat_check", "drive_thru", "happy_hour", "best_nights", 
    "noise_level", "bike_parking", "dogs_allowed", "good_for_kids", "good_for_meal", 
    "good_for_dancing", "outdoor_seating", "business_parking", "by_appointment_only", 
    "restaurants_attire", "restaurants_take_out", "restaurants_delivery", 
    "wheelchair_accessible", "business_accepts_bitcoin", "restaurants_price_range", 
    "restaurants_reservations", "restaurants_table_service", "restaurants_good_for_groups", 
    "business_accepts_credit_cards", "hours_monday", "hours_tuesday", "hours_wednesday", 
    "hours_thursday", "hours_friday", "hours_saturday", "hours_sunday","text","date"
]

data = data.drop(columns=columns_to_drop)


#%%

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

data['sunshine_duration'] = data['sunshine_duration'].round(2)
data['daylight_duration'] = data['daylight_duration'].round(2)

#%%

# Wettergruppen definieren
weather_groups = {
    "Clear": ["Clear sky", "Mainly clear", "Partly cloudy"],
    "Cloudy": ["Overcast"],
    "Fog": ["Fog", "Rime fog"],
    "Rain": ["Light drizzle", "Moderate drizzle", "Dense drizzle", 
             "Freezing drizzle", "Dense freezing drizzle", 
             "Slight rain", "Moderate rain", "Heavy rain", 
             "Freezing rain", "Heavy freezing rain", 
             "Slight rain showers", "Moderate rain showers", "Heavy rain showers"],
    "Snow": ["Slight snow fall", "Moderate snow fall", "Heavy snow fall", 
             "Snow grains", "Slight snow showers", "Heavy snow showers"],
    "Thunderstorm": ["Thunderstorm", "Thunderstorm with slight hail", "Thunderstorm with heavy hail"]
}

# Funktion zur Gruppierung der Wetterbeschreibung
def map_weather_to_group(description):
    for group, descriptions in weather_groups.items():
        if description in descriptions:
            return group
    return "Other"  # Falls ein Wert nicht zugeordnet werden kann

# Anwendung der Funktion auf die Spalte
data['weather_description'] = data['weather_description'].apply(map_weather_to_group)

data.rename(columns={'stars_x': 'stars_review'}, inplace=True)



#%%

# Anpassen möglicher Werte über 5 und unter 1

analyze_column('FoodnDrinks_Rating')

#%%

data['FoodnDrinks_Rating'] = data['FoodnDrinks_Rating'].apply(lambda x: 5 if x > 5 else x)
data['FoodnDrinks_Rating'] = data['FoodnDrinks_Rating'].apply(lambda x: 1 if x < 1 else x)
analyze_column('FoodnDrinks_Rating')

#%%

analyze_column('Service_Rating')

#%%

data['Service_Rating'] = data['Service_Rating'].apply(lambda x: 5 if x > 5 else x)
data['Service_Rating'] = data['Service_Rating'].apply(lambda x: 1 if x < 1 else x)
analyze_column('Service_Rating')


#%%

analyze_column('Atmosphere_Rating')

#%%

data['Atmosphere_Rating'] = data['Atmosphere_Rating'].apply(lambda x: 5 if x > 5 else x)
data['Atmosphere_Rating'] = data['Atmosphere_Rating'].apply(lambda x: 1 if x < 1 else x)
analyze_column('Atmosphere_Rating')


#%%

analyze_column('Price_Rating')

#%%

data['Price_Rating'] = data['Price_Rating'].apply(lambda x: 5 if x > 5 else x)
data['Price_Rating'] = data['Price_Rating'].apply(lambda x: 1 if x < 1 else x)
analyze_column('Price_Rating')

#%%

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data.to_csv(f"C:/Users/lukas/Desktop/DSMA/Working Data/Review_Weather_Data_Clean_{timestamp}.csv", index=False)