# DATA SPLITTING
# Target: Train=70%, Val=15%, Test=15%

# Langkah 1: Pisahkan Test (15%) dari total data
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)

# Langkah 2: Pisahkan Train (70%) dan Val (15%) dari sisa data
# Hitungan: 0.15 / 0.85 (sisa) ~= 0.176
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
)

print(f"Total Data: {len(X_scaled)}")
print(f"Train Set : {X_train.shape[0]} ({X_train.shape[0]/len(X_scaled):.1%})")
print(f"Val Set   : {X_val.shape[0]} ({X_val.shape[0]/len(X_scaled):.1%})")
print(f"Test Set  : {X_test.shape[0]} ({X_test.shape[0]/len(X_scaled):.1%})")
print("✅ Data Splitting Selesai.")

# DATA BALANCING
# Kita gunakan teknik Class Weighting (bukan oversampling) untuk menjaga keaslian data
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: weights[i] for i in range(len(weights))}

print(f"Class Weights: {class_weight_dict}")