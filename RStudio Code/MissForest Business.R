# Installiere und lade das missForest-Paket
install.packages("missForest")
library(missForest)


# Beispiel: Lade das Datenset (ersetze dies mit deinem tatsächlichen Datensatz)
data <- read.csv(choose.files(), stringsAsFactors = TRUE)
summary(data)


# Stelle sicher, dass alle fehlenden Werte als NA erkannt werden
data[data == ""] <- NA  # Leere Strings als NA behandeln (falls vorhanden)
data[data == NULL] <- NA # NULL-Werte als NA behandeln (falls vorhanden)

# Verwende missForest zur Imputation der fehlenden Werte
imputed_data <- missForest(data)

# Das imputierte Datenset ist in imputed_data$ximp gespeichert
completed_data <- imputed_data$ximp

# Überprüfe das imputierte Datenset
summary(completed_data)

# Optional: Speichere das imputierte Datenset
# write.csv(completed_data, "imputierte_datei.csv")

# Berechne und drucke den Prozentsatz der fehlenden Werte im ursprünglichen Datensatz
missing_percentage <- colMeans(is.na(data))
print(missing_percentage)




# Spalten mit mehr als 53 Kategorien finden
high_cardinality_cols <- sapply(data, function(col) is.factor(col) && nlevels(col) > 53)

# Datensatz ohne diese Spalten
reduced_data <- data[, !high_cardinality_cols]

# MissForest auf reduzierte Daten anwenden
imputed_data <- missForest(reduced_data)

# Ursprüngliche Spalten wieder hinzufügen
imputed_data$ximp <- cbind(imputed_data$ximp, data[, high_cardinality_cols])



final_data <- as.data.frame(imputed_data$ximp)


# Reihenfolge der Spalten von 'data' übernehmen
final_data <- final_data[, colnames(data)]

# Überprüfen, ob die Spaltenreihenfolge übereinstimmt
all(colnames(final_data) == colnames(data))  # Sollte TRUE ausgeben

final_data$restaurants_price_range <- round(final_data$restaurants_price_range)



# Aktuelles Datum und Uhrzeit im gewünschten Format
timestamp <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")

# Datei speichern mit Datum und Uhrzeit im Namen
write.csv(final_data, paste0("C:/Users/lukas/Desktop/DSMA/Working Data/Business_Data_Sushi_Bars_Clean_Filled_", timestamp, ".csv"), row.names = FALSE)

