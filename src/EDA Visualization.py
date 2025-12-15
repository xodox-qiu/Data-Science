# VISUALIZATION

# 1. Visualisasi Distribusi Kelas Target
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title('Distribusi Kelas Target (0=Non-Toxic, 1=Toxic)')
plt.savefig('images/viz1_class_distribution.png') # Save
plt.show()

# 2. Visualisasi Heatmap Korelasi (Fitur Terpilih)
plt.figure(figsize=(10, 8))
# Menggabungkan X dan y sementara untuk korelasi
df_corr = X_final.copy()
df_corr['Target'] = y.values
corr = df_corr.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False) # Annot false agar tidak penuh angka
plt.title('Heatmap Korelasi (30 Fitur Terpilih)')
plt.savefig('images/viz2_correlation.png') # Save
plt.show()

# 3. Visualisasi Training History (Loss) untuk Deep Learning
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Deep Learning: Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('images/viz3_dl_loss.png') # Save
plt.show()

print("✅ Visualisasi selesai dan disimpan di folder 'images/'.")