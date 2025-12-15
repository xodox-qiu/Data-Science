# DATA TRANSFORMATION
scaler = StandardScaler()

# Melakukan scaling pada X_final
X_scaled_array = scaler.fit_transform(X_final)
X_scaled = pd.DataFrame(X_scaled_array, columns=selected_features)

print("Beberapa data setelah scaling:")
print(X_scaled.iloc[:3, :5]) # Print 3 baris pertama, 5 kolom pertama

# SIMPAN PROCESSED DATA

processed_dataset = pd.concat([X_scaled, y], axis=1)

processed_dataset.to_csv('data/toxicity_processed_data.csv', index=False)
print("Berhasil! File 'toxicity_processed_data.csv' siap digunakan untuk modelling.")