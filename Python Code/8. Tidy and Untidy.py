import pandas as pd
from datetime import datetime

data = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Working Data/Data_Ready_Not_Tidy_2025-01-01_15-24-57.csv")

def split_column(column_name):
    unique_values = data[column_name].dropna().unique()
    for value in unique_values:
        data[f'{column_name}_{value}'] = data[column_name].apply(lambda x: value in str(x))
    data.drop(column_name, axis=1, inplace=True)
    
    
    
def reverse_split_columns(column_prefix):
    # Suche alle Spalten, die mit dem angegebenen Präfix beginnen
    columns_to_merge = [col for col in data.columns if col.startswith(column_prefix)]
    
    # Kombiniere die True/False-Spalten zurück zu einer einzelnen Spalte
    data[column_prefix] = data[columns_to_merge].apply(lambda row: ', '.join([col.split('_')[-1] for col in columns_to_merge if row[col]]), axis=1)
    
    # Lösche die einzelnen binären Spalten
    data.drop(columns=columns_to_merge, inplace=True)


#%%

split_column('alcohol')
reverse_split_columns("alcohol")


#%%

split_column('noise_level')
reverse_split_columns("noise_level")

#%%

split_column('restaurants_attire')
reverse_split_columns("restaurants_attire")


#%%

split_column('weather_description')
reverse_split_columns("weather_description")

#%%

split_column("state")
reverse_split_columns("state")

#%%

data = data.drop(columns=["business_id"])

#%%

data = data.drop(columns=["weather_description"])


#%%

for column in data.columns:
    if data[column].dtype == 'bool':
        data[column] = data[column].astype(int)


#%%

average_stars = data['stars_review'].mean()
data['stars_above_average'] = (data['stars_review'] > average_stars).astype(int)

#%%

columns = ['stars_above_average'] + [col for col in data.columns if col != 'stars_above_average']
data = data[columns]

#%%


# Angenommen, dein DataFrame heißt 'df'
data['noise_level'] = data['noise_level'].replace({'loud, loud': 'very_loud'})

data['alcohol'] = data['alcohol'].replace({'bar': 'full_bar', 'wine': 'beer_and_wine'})

#%%

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data.to_csv(f"C:/Users/lukas/Desktop/DSMA/Working Data/Combined_Not_Tidy_Data_{timestamp}.csv", index=False)

#%%

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
data.to_csv(f"C:/Users/lukas/Desktop/DSMA/Working Data/Combined_Tidy_Data_{timestamp}.csv", index=False)




