import pandas as pd
import os

# Daten einlesen
data_1 = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Processed Data/Sushi Without NaNs/Processed_Review_Data_GPT4_Batch_Sushi_Bars_1.csv")
data_2 = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Processed Data/Sushi Without NaNs/Processed_Review_Data_GPT4_Batch_Sushi_Bars_2.csv")
data_3 = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Processed Data/Sushi Without NaNs/Processed_Review_Data_GPT4_Batch_Sushi_Bars_3.csv")
data_4 = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Processed Data/Sushi Without NaNs/Processed_Review_Data_GPT4_Batch_Sushi_Bars_4.csv")

# DataFrames kombinieren
combined_data = pd.concat([data_1, data_2, data_3, data_4], ignore_index=True)

# Kombinierte Daten speichern
output_file = os.path.join("C:/Users/lukas/Desktop/DSMA/Working Data", "Processed_Review_Data_without_NaNs.csv")
combined_data.to_csv(output_file, index=False)

print("Files successfully combined and saved as 'Processed_Review_Data_without_NaNs.csv'.")
