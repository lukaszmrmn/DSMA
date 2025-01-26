import openai
import pandas as pd
import json
from datetime import datetime, timedelta


# Daten laden in 50.000 Batches
data = pd.read_csv("C:/Users/lukas/Desktop/DSMA/Original Data/Review Data Sushi Bars.csv")
data = data.iloc[100000:150000].reset_index(drop=True)


#%%

# API-Client initialisieren
client = openai.OpenAI(api_key="API-KEY REMOVED")

# Token-Zähler
total_tokens_used = 0

# Funktion zur Analyse von Batch-Reviews
def analyze_batch_reviews(reviews_batch):
    global total_tokens_used  # Zugriff auf die globale Variable
    # Formatiere die Reviews als Liste
    review_texts = "\n\n".join(
        [f"Review {i+1}: {review}" for i, review in enumerate(reviews_batch)]
    )
    messages = [
        {"role": "system", "content": "You are an assistant that analyzes restaurant reviews. Ensure all ratings are integers between 1 and 5."},
        {"role": "user", "content": f"""
            Analyze the following reviews and rate them on a scale from 1 to 5 for the categories 
            FoodnDrinks, Service, Atmosphere, and Price. If no information is available for a category, 
            assign NaN to that category. 
            {review_texts}
            Ensure the output is always in this exact format:
            [
                {{"FoodnDrinks": <1-5 or NaN>, "Service": <1-5 or NaN>, "Atmosphere": <1-5 or NaN>, "Price": <1-5 or NaN>}},
                ...
            ]
        """}
    ]
    try:
        # API-Anfrage
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.5,
            max_tokens=1000
        )
        # API-Antwort ausgeben mit der aktuellen Zeit
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"API response at {current_time} - Batch {i // batch_size + 1}/{len(data) // batch_size}")
        
        # Tokens summieren
        total_tokens_used += response.usage.total_tokens
        
        
        # Extrahiere die Antwort
        result_content = response.choices[0].message.content
        result_json = json.loads(result_content)
        return result_json
    except Exception as e:
        print(f"Error analyzing batch: {e}")
        return None

# Verarbeitung der Daten in Batches
batch_size = 5
results = []

start_time = datetime.now()


for i in range(0, len(data), batch_size):
    # Slice des aktuellen Batches
    batch = data.iloc[i:i + batch_size]
    batch_reviews = batch["text"].tolist()
    
    # Analysiere den Batch
    batch_results = analyze_batch_reviews(batch_reviews)
    
    if batch_results:
        # Speichere die Ergebnisse für jede Zeile im Batch
        for idx, review_result in enumerate(batch_results):
            results.append({
                "index": batch.index[idx],
                "FoodnDrinks_Rating": review_result.get("FoodnDrinks"),
                "Service_Rating": review_result.get("Service"),
                "Atmosphere_Rating": review_result.get("Atmosphere"),
                "Price_Rating": review_result.get("Price")
            })
    else:
        # Leere Ergebnisse für den Batch hinzufügen
        for idx in range(len(batch_reviews)):
            results.append({
                "index": batch.index[idx],
                "FoodnDrinks_Rating": None,
                "Service_Rating": None,
                "Atmosphere_Rating": None,
                "Price_Rating": None
            })




# Ergebnisse in ein DataFrame konvertieren
results_df = pd.DataFrame(results).set_index("index")

# Ergebnisse mit dem Original-DataFrame zusammenführen
data = data.join(results_df)

# Ergebnisse anzeigen
print(data)

# Gesamte verbrauchte Tokens ausgeben
print(f"Total tokens used: {total_tokens_used}")

# Gesamtverarbeitungszeit berechnen
end_time = datetime.now()
total_time = end_time - start_time
print(f"Total execution time: {total_time}")

# Daten in eine CSV-Datei exportieren
data.to_csv("C:/Users/lukas/Desktop/DSMA/Python Code/Data Output/Processed_Review_Data_GPT4_Batch_Sushi_Bars_3_with_NaN.csv", index=False)
