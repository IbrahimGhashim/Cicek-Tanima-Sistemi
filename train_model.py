import json
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

# =========================
# Temel ayarlar
# =========================

# Modelin kabul edeceği görsel boyutu belirlenir
IMG_SIZE = (224, 224)

# Eğitim sırasında her adımda kaç görsel kullanılacağı belirlenir
BATCH_SIZE = 32

# İlk eğitim ve fine tuning için epoch sayıları
EPOCHS = 12
FINE_TUNE_EPOCHS = 8
SEED = 123

# =========================
# Veri setini indirme
# =========================

# Çiçek görsellerinden oluşan veri setinin bağlantısı
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# Veri seti indirilir ve otomatik olarak açılır
data_dir = tf.keras.utils.get_file(
    "flower_photos",
    origin=dataset_url,
    untar=True
)

# İndirilen veri setinin içindeki asıl görsel klasörüne geçilir
data_dir = pathlib.Path(data_dir) / "flower_photos"

print("Veri seti konumu:", data_dir)

# =========================
# Klasör kontrolü
# =========================

# Veri setinde bulunması gereken çiçek sınıfları
class_folder_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# Her sınıf klasörünün var olup olmadığı kontrol edilir
for folder_name in class_folder_names:
    folder_path = data_dir / folder_name

    if not folder_path.exists():
        raise FileNotFoundError(f"Eksik klasör bulundu: {folder_path}")

# =========================
# Eğitim ve doğrulama verilerini hazırlama
# =========================

# Veri setinin %80'i eğitim, %20'si doğrulama için ayrılır
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Veri setindeki sınıf isimleri alınır
class_names = train_ds.class_names
num_classes = len(class_names)

print("Sınıflar:", class_names)
print("Sınıf sayısı:", num_classes)

# Sınıf isimleri daha sonra arayüzde kullanılmak üzere dosyaya kaydedilir
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)

# =========================
# Veri performansını artırma
# =========================

#TensorFlow’un otomatik performans ayarlamak için
AUTOTUNE = tf.data.AUTOTUNE

# Eğitim sürecini hızlandırmak için veriler belleğe alınır,
# karıştırılır ve modelin daha hızlı okuyabileceği şekilde hazırlanır
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =========================
# Veri artırma işlemleri
# =========================

# Veri artırma, modelin sadece eğitim resimlerini ezberlemesini azaltır.
# Böylece farklı açılardan veya farklı ışıkta çekilen görsellerde daha iyi sonuç verebilir.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.20),
    layers.RandomZoom(0.20),
    layers.RandomContrast(0.15),
])

# =========================
# Hazır modelin yüklenmesi
# =========================

# EfficientNetB0 hazır eğitilmiş bir görüntü sınıflandırma modelidir.
# include_top=False ile modelin eski sınıflandırma katmanı çıkarılır.
# Böylece kendi 5 çiçek sınıfımız için yeni bir çıkış katmanı eklenebilir.
base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# İlk aşamada hazır modelin ana katmanları dondurulur.
# Bu aşamada sadece bizim eklediğimiz yeni sınıflandırma katmanı eğitilir.
base_model.trainable = False

# =========================
# Modelin oluşturulması
# =========================

# Modelin giriş boyutu belirlenir
inputs = tf.keras.Input(shape=(224, 224, 3))

# Eğitim sırasında görsellere küçük değişiklikler uygulanır
x = data_augmentation(inputs)

# Görseller hazır modelden geçirilir
x = base_model(x, training=False)

# Özellik haritası tek boyutlu hale getirilir
x = layers.GlobalAveragePooling2D()(x)

# Dropout, modelin ezber yapmasını azaltmaya yardımcı olur
x = layers.Dropout(0.4)(x)

# Softmax katmanı her çiçek sınıfı için olasılık değeri üretir
outputs = layers.Dense(num_classes, activation="softmax")(x)

# Giriş ve çıkış katmanları birleştirilerek model oluşturulur
model = tf.keras.Model(inputs, outputs)

# =========================
# İlk derleme işlemi
# =========================

# Modelin eğitim ayarları yapılır
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Modelin katman yapısı ekrana yazdırılır
model.summary()

# =========================
# İlk eğitim aşaması
# =========================

print("İlk eğitim başlıyor...")

# İlk eğitimde hazır modelin ana katmanları sabit kalır
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =========================
# Fine tuning aşaması
# =========================

print("Fine tuning başlıyor...")

# Fine tuning aşamasında hazır modelin son katmanları tekrar eğitilebilir hale getirilir
base_model.trainable = True

# İlk katmanlar sabit bırakılır, sadece son 40 katman eğitilir.
# Böylece model genel bilgilerini korurken çiçek veri setine daha iyi uyum sağlar.
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Fine tuning sırasında düşük öğrenme oranı kullanılır.
# Bunun amacı modelin öğrendiği bilgileri bozmadan küçük iyileştirmeler yapmaktır.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Model ikinci kez, bu sefer fine tuning için eğitilir
fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS
)

# =========================
# Modelin değerlendirilmesi
# =========================

# Doğrulama verisi üzerinde modelin başarısı ölçülür
loss, accuracy = model.evaluate(val_ds)

print("Doğrulama kaybı:", loss)
print("Doğrulama başarısı:", accuracy)

# =========================
# Modeli kaydetme
# =========================

# Eğitilen model daha sonra arayüzde kullanılmak üzere kaydedilir
model.save("flower_model.keras")

print("Model kaydedildi: flower_model.keras")
print("Sınıf isimleri kaydedildi: class_names.json")