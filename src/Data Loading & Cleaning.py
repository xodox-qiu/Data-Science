# DATA LOADING & CLEANING
print("Sedang mengambil dataset dari UCI Repository...")
from ucimlrepo import fetch_ucirepo
toxicity = fetch_ucirepo(id=728)

# Mengambil fitur dan target
X_raw = toxicity.data.features
y_raw = toxicity.data.targets

# Menggabungkan Fitur dan Target menjadi satu DataFrame
raw_dataset = pd.concat([X_raw, y_raw], axis=1)

# Simpan ke file
raw_dataset.to_csv('data/toxicity_raw_data.csv', index=False)
print("Berhasil! File tersimpan sebagai 'toxicity_raw_data.csv'.")

print(f"Dataset Asli: {X_raw.shape[0]} Baris, {X_raw.shape[1]} Fitur")

print("\n[Tabel] 5 Baris Pertama Data MENTAH (Sebelum Cleaning):")
display(X_raw.head()) # Menampilkan tabel awal

# Cek Missing Values & Cleaning
print("\n--- Proses Cleaning ---")
missing_count = X_raw.isnull().sum().sum()
if missing_count > 0:
    print(f"Ditemukan {missing_count} missing values.")
    print("Melakukan imputasi (mengisi nilai kosong) dengan rata-rata...")
    X_raw.fillna(X_raw.mean(), inplace=True)
    print("Imputasi selesai.")
else:
    print("Data sudah bersih (Tidak ada missing values).")

# --- MENAMPILKAN DATA SETELAH CLEANING ---
print("\n[Tabel] 5 Baris Pertama Data BERSIH (Setelah Cleaning):")
display(X_raw.head()) # Menampilkan tabel setelah dibersihkan

# Encoding Target (Mengubah 'Toxic'/'NonToxic' menjadi 1/0)
print("\n--- Encoding Target ---")
le = LabelEncoder()
# Mengambil kolom pertama dari y_raw karena y_raw adalah DataFrame
y_encoded = le.fit_transform(y_raw.iloc[:, 0])
y = pd.Series(y_encoded, name='Target')

# Simpan mapping kelas untuk referensi nanti
class_names = le.classes_
print(f"Mapping Kelas: {class_names} -> [0, 1]")