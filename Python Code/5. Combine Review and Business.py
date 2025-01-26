import pandas as pd
from datetime import datetime


# Einlesen der beiden CSV-Dateien
review_weather_data = pd.read_csv('C:/Users/lukas/Desktop/DSMA/Working Data/Review_Weather_Data_Clean_2024-12-29_23-18-40.csv')
business_data = pd.read_csv('C:/Users/lukas/Desktop/DSMA/Working Data/Business_Data_Sushi_Bars_Clean_Filled_Minimized_2024-12-29_23-16-46.csv')

# Kombinieren der beiden DataFrames anhand der 'business_id'
combined_data = pd.merge(review_weather_data, business_data, on='business_id', how='left')



timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Speichern des kombinierten DataFrames als neue CSV-Datei mit Zeitstempel im Dateinamen
combined_data.to_csv(f'C:/Users/lukas/Desktop/DSMA/Working Data/Combined_Data_{timestamp}.csv', index=False)
