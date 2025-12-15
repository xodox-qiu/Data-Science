# EVALUATION & COMPARISON

# 1. Pastikan Dictionary Model sudah siap
models_dict = {
    'Baseline (LogReg)': model_baseline,
    'Advanced (RF)': model_rf,
    'Deep Learning (MLP)': model_dl
}

# Container hasil
final_metrics = []

print("=== HASIL EVALUASI PADA DATA TEST (15%) ===")

for name, model in models_dict.items():
    print(f"\n>> Mengevaluasi Model: {name}...")

    # 2. Lakukan Prediksi
    if name == 'Deep Learning (MLP)':
        # Output DL adalah probabilitas, ubah jadi 0/1
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    else:
        # Output Sklearn langsung kelas 0/1
        y_pred = model.predict(X_test)

    # 3. Hitung Metrik
    acc = accuracy_score(y_test, y_pred)

    # Gunakan average='weighted' agar F1-Score lebih adil
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    final_metrics.append({
        'Model': name,
        'Accuracy': acc,
        'F1-Score': f1
    })

    # Print Laporan Detail
    print(classification_report(y_test, y_pred, zero_division=0))

# 4. Membuat Tabel DataFrame
df_metrics = pd.DataFrame(final_metrics)

print("\n=== TABEL PERBANDINGAN AKHIR ===")
# --- PERBAIKAN DI SINI ---
# Kita format spesifik per kolom agar kolom 'Model' (Text) tidak error
display(df_metrics.style.format({
    'Accuracy': '{:.4f}',
    'F1-Score': '{:.4f}'
}))

# 5. Visualisasi
plt.figure(figsize=(10, 6))
df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")

ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Perbandingan Performa Model (Weighted F1)", fontsize=14)
plt.ylim(0, 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

# Label Angka di atas batang
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

plt.savefig('images/viz4_model_comparison.png')
plt.show()