# NLP-EMOTION-DETECTION

# Analisis Klasifikasi Emosi di Media Sosial dengan Deep Learning

Reaksi, ulasan, tanggapan, komentar, serta kritik yang disampaikan di platform-platform media sosial sangat penting sebagai indikator bagi pembuat konten. Secara umum, reaksi ini dapat dikategorikan sebagai **positif** atau **negatif**, atau dibagi menjadi indikator yang lebih spesifik, seperti **sadness, joy, love, anger, fear, surprise**.

Dengan indikator di atas, tugas untuk menganalisis reaksi ini dapat diartikan sebagai tugas **klasifikasi**, di mana setiap kategori merepresentasikan jenis emosi atau tanggapan. Analisis reaksi ini sangat berguna bagi content creator dan platform streaming untuk menilai seberapa baik konten mereka diterima oleh audiens serta merancang strategi untuk meningkatkan kualitas konten di masa mendatang.

## Problem Statements

Berdasarkan latar belakang di atas, berikut ini batasan masalah yang dapat diselesaikan dengan proyek ini:

- **Bagaimana cara membangun sebuah model deep learning yang optimal untuk dapat melakukan klasifikasi emosi?**
- **Bagaimanakah bentuk arsitektur model terbaik yang bisa digunakan untuk memprediksi emosi?**

## Goals

1. Membuat pra-pemrosesan data sebaik mungkin untuk pemrosesan model lebih lanjut.
2. Membangun model deep learning untuk memprediksi sentimen yang ada pada dataset.

## Solution Statements

Solusi yang dapat dilakukan sebagai berikut:

- Membuat model deep learning berbasis **LSTM** untuk klasifikasi sentimen.
- Menggunakan **Classification Report** untuk mengevaluasi model.

## Data Understanding

Dataset yang digunakan dapat diakses menggunakan Kaggle di link berikut:
[https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data)

Pada berkas yang diunduh berisi 1 file .txt yang total berisi **311.158** teks dan label dengan isi sebagai berikut:

- **Text**: kalimat/sentimen/pendapat/kritik yang ditulis oleh individu.
- **Label**: perasaan yang dirasakan oleh individu.

## Langkah-langkah Pra-pemrosesan Data

1. **Mendownload dataset** dari Kaggle.
2. Membaca dataset yang telah didownload ke DataFrame.
3. Menampilkan informasi dari dataset.
4. Mengecek dan menangani missing value di dataset.
5. Mengecek sample text yang ada di DataFrame.

Pada proyek ini dataset di-download melalui Kaggle dari link berikut:  
[https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data).

### Pra-Pemrosesan Data

1. **Membaca dataset ke DataFrame pandas**  
   Pada bagian ini akan digunakan fungsi `pandas.read_csv()` untuk membaca berkas .txt, karena berkas bersifat .txt, tidak memiliki header, dan pemisah antara variabel text dan label adalah tanda koma.

2. **Menampilkan informasi dari dataset**  
   Pada bagian ini akan digunakan fungsi `shape()` dan `value_counts()` untuk mengetahui jumlah dataset dan distribusi dari label.

3. **Mengecek missing value, duplicate, dan null**  
   Pada bagian ini digunakan fungsi `isnull().sum()` untuk tiap DataFrame. Saat dicek, tidak ditemukan adanya missing value.

### Data Preparation

- Menghapus angka yang ada di variabel text.
- Menghapus tanda baca yang ada di variabel text.
- Merubah tiap kata di variabel text menjadi bentuk dasar (lemmatization).
- Merubah kata menjadi lowercase pada variabel text.
- Menghapus stopwords pada variabel text.
- Merubah variabel label menjadi numerik.

Pada bagian ini, teks yang ada akan diproses untuk menghapus, merubah, dan lowercase. Untuk melakukan itu digunakan beberapa library yaitu **re**, **nltk**, dan **pandas**. Dibuat sebuah fungsi baru yang bernama `clean_text` yang tugasnya adalah memanggil fungsi yang ada pada 3 library.

- Lowercase teks menggunakan pandas dengan fungsi `lower()`.
- Menghapus angka dan tanda baca menggunakan **re** (regular expression).
- Merubah kata menjadi bentuk dasar menggunakan **nltk.stem.WordNetLemmatizer**.
- Menghapus stopwords menggunakan **nltk.corpus.stopwords**.

Jika dicek perubahan yang dilakukan dengan fungsi `head()`, hasilnya akan terlihat.

### Mengubah Label Menjadi Numerik

Pada bagian ini, variabel label akan diubah dari kategorikal menjadi numerik. Variabel label memiliki 6 kategori yaitu **anger, fear, joy, love, sadness, dan surprise**. Kategori ini akan diubah menjadi numerik (0-5) dengan menggunakan **LabelEncoder** seperti di tabel berikut:

Tabel 1. Encoder kategori

| label   | label_encode |
| ------- | ------------ |
| anger   | 0            |
| fear    | 1            |
| joy     | 2            |
| love    | 3            |
| sadness | 4            |
| surprise| 5            |

## Modeling

**Deep Learning** adalah teknik yang memungkinkan model komputasi terdiri dari banyak layer proses. Layer ini mempelajari representasi data dengan level abstraksi yang beragam. **Deep Learning** menemukan struktur yang menarik dengan dataset besar menggunakan algoritma backpropagation, yang mengindikasikan bagaimana mesin harus mengganti parameter untuk menghitung tiap layer dari representasi layer sebelumnya.

### Tahapan Umum Cara Kerja Deep Learning

- Data akan masuk ke layer pertama.
- Di layer pertama terdapat sejumlah neuron yang memproses informasi yang diberikan.
- Setiap neuron merepresentasikan informasi yang diproses, misalnya dengan dataset yang dipakai, neuron 1 bisa saja merepresentasikan sentimen **anger**.
- Informasi akan disalurkan ke channel penghubung. Sebelum disalurkan, dihitung nilai weight & bias yang diterapkan di **activation function**.
- Hasil dari activation function akan menentukan apakah neuron di layer selanjutnya dapat diaktifkan.
- Setiap neuron yang diaktifkan akan meneruskan informasi ke layer selanjutnya hingga layer terakhir.

Pada proyek ini, karena bersifat **classification** dan dataset adalah teks, digunakan deep learning dengan tipe **Bidirectional Recurrent-Neural-Network (BRNN)**. Model ini mempelajari pola jangka panjang dan mengingat informasi di layer. BRNN menarik karena terdapat 2 RNN yang independen, satu RNN diberikan input dari layer pertama ke terakhir, dan satu lagi terbalik. Keluaran dari 2 RNN digabungkan tiap kali 1 perulangan. Tipe ini memungkinkan jaringan saraf tiruan untuk diberikan informasi **backward** dan **forward** setiap perulangan.

### Arsitektur Model yang Digunakan

- **Menggunakan tf.keras.layers.Embedding dengan input_dim=len(vectorize_layer.get_vocabulary()) dan output_dim=32** untuk merepresentasikan setiap kata dalam vektor dengan   dimensi 32.
- **Hidden layer (Bidirectional LSTM)** dengan 64 neuron.
- **Layer Output (Dense)** dengan 6 neuron (jumlah kategori pada label).

## Evaluasi
Model deep learning yang bertipe **classification** dievaluasi berdasarkan **accuracy, precision, recall, dan f1-score**. Metrik-metrik ini dihitung berdasarkan nilai **True Positive (TP)**, **True Negative (TN)**, **False Positive (FP)**, dan **False Negative (FN)**.

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-Score** = 2 \* (Precision \* Recall) / (Precision + Recall)

Dengan ini, model dapat dioptimalkan untuk melakukan klasifikasi sentimen secara akurat dan efisien. Berikut plot akurasi model
![image](https://github.com/user-attachments/assets/a3987215-a661-4f7f-b384-9b5fb30cf847)

gambar 1. Akurasi model

Dapat dilihat dari gambar 1 bahwa setelah epochs ke 5 model membuat model yang Good Fit
Penulis juga menguji model dengan data test yang sebelumnya sudah dipisahkan dengan hasil seperti berikut tabel 10.

Gambar 2. Classification Report
![image](https://github.com/user-attachments/assets/72035ac1-6c69-44d9-92ce-5b653cd6d951)

Model Performance
Model bekerja dengan baik dalam memprediksi keenam label emosi, dengan nilai precision, recall, dan f1-score yang hampir semuanya berada di atas 70%. Akurasi model pada data uji mencapai 93%. Berikut adalah penjelasan untuk precision dan recall secara spesifik:

Precision

Label 0 (Kelas 0): Dari 100% prediksi untuk kelas ini, 95% di antaranya benar-benar kelas 0.

Label 1 (Kelas 1): Dari 100% prediksi untuk kelas ini, 95% di antaranya benar-benar kelas 1.

Label 2 (Kelas 2): Dari 100% prediksi untuk kelas ini, 83% di antaranya benar-benar kelas 2.

Label 3 (Kelas 3): Dari 100% prediksi untuk kelas ini, 97% di antaranya benar-benar kelas 3.

Label 4 (Kelas 4): Dari 100% prediksi untuk kelas ini, 88% di antaranya benar-benar kelas 4.

Label 5 (Kelas 5): Dari 100% prediksi untuk kelas ini, 82% di antaranya benar-benar kelas 5.

Recall

Label 0 (Kelas 0): Dari semua data yang benar-benar kelas 0, 99% berhasil diprediksi dengan benar.

Label 1 (Kelas 1): Dari semua data yang benar-benar kelas 1, 95% berhasil diprediksi dengan benar.

Label 2 (Kelas 2): Dari semua data yang benar-benar kelas 2, 82% berhasil diprediksi dengan benar.

Label 3 (Kelas 3): Dari semua data yang benar-benar kelas 3, 91% berhasil diprediksi dengan benar.

Label 4 (Kelas 4): Dari semua data yang benar-benar kelas 4, 90% berhasil diprediksi dengan benar.

Label 5 (Kelas 5): Dari semua data yang benar-benar kelas 5, 71% berhasil diprediksi dengan benar.

F1-Score

F1-score menunjukkan keseimbangan antara precision dan recall untuk setiap label:

Label 0: 0.97

Label 1: 0.95

Label 2: 0.83

Label 3: 0.94

Label 4: 0.89

Label 5: 0.76

Rata-Rata dan Akurasi

Akurasi: 93%

Macro Average (rata-rata tidak berbobot) untuk precision, recall, dan f1-score masing-masing sebesar 0.90, 0.88, dan 0.89.
Weighted Average (rata-rata berbobot sesuai dengan jumlah sampel pada setiap kelas) masing-masing sebesar 0.93 untuk precision, recall, dan f1-score.






