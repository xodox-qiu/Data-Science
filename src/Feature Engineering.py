# FEATURE ENGINEERING
print("Melakukan Seleksi Fitur...")

selector = SelectKBest(score_func=f_classif, k=30)
X_selected = selector.fit_transform(X_raw, y)

# Mendapatkan nama fitur yang terpilih
mask = selector.get_support()
selected_features = X_raw.columns[mask]

# Membuat DataFrame baru dengan fitur terpilih
X_final = pd.DataFrame(X_selected, columns=selected_features)

print(f"Dimensi Awal: {X_raw.shape}")
print(f"Dimensi Setelah Seleksi: {X_final.shape}")
print(f"Fitur Terpilih: {list(selected_features[:5])} ...") # Print 5 fitur pertama