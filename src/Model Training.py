# MODEL 1 - BASELINE (Logistic Regression)
print("Training Baseline Model...")
model_baseline = LogisticRegression(random_state=42, class_weight='balanced')
model_baseline.fit(X_train, y_train)

# Simpan Model
joblib.dump(model_baseline, 'models/model_baseline.pkl')
print("✅ Baseline Model Trained & Saved.")

# MODEL 2 - ADVANCED (Random Forest)
print("Training Advanced Model...")
# Menggunakan n_estimators=100 dan max_depth dibatasi agar tidak overfitting di data kecil
model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model_rf.fit(X_train, y_train)

# Simpan Model
joblib.dump(model_rf, 'models/model_advanced.pkl')
print("✅ Advanced Model Trained & Saved.")

# MODEL 3 - DEEP LEARNING (MLP)
print("Training Deep Learning Model...")

# Arsitektur Model
# Karena data sedikit, kita gunakan Dropout yang cukup besar untuk mencegah overfitting
model_dl = Sequential([
    Input(shape=(X_train.shape[1],)),       # Input Layer (30 fitur)
    Dense(64, activation='relu'),           # Hidden Layer 1
    Dropout(0.5),
    Dense(32, activation='relu'),           # Hidden Layer 2
    Dropout(0.3),
    Dense(1, activation='sigmoid')          # Output Layer (Binary)
])

model_dl.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training dengan Early Stopping
history = model_dl.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,             # Maksimum epoch
    batch_size=16,
    class_weight=class_weight_dict,
    verbose=0,              # Silent training agar tidak penuh outputnya
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
)

# Simpan Model
model_dl.save('models/model_toxicity_dl.keras')
print("✅ Deep Learning Model Trained & Saved.")