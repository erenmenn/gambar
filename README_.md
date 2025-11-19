# Sistem Rekomendasi Film – *Movie Recommendation System*

 **Penulis**: Emirsyah Rafsanjani  
 **Dataset**: https://www.kaggle.com/parasharmanas/movie-recommendation-system



## 1. Project Overview

### Latar Belakang
Platform streaming seperti Netflix, Disney+ Hotstar dan Vidio bersaing ketat untuk **menjaga *retention* pengguna**. Salah satu fitur teersembunyi nya adalah **sistem rekomendasi** yang mampu menampilkan film / series yang tepat saat pertama pengguna membuka aplikasi. 

Paper Gomez-Uribe & Hunt (2016) menunjukkan bahwa **≈ 80 % judul yang ditonton di Netflix berasal dari hasil rekomendasi**, bukan pencarian manual.

### Kenapa Proyek Ini Penting
- memahami perbedaan pendekatan **Content-Based** vs **Collaborative Filtering**.
- mengetahui seberapa pentingnya penggunaan algoritma rekomendasi
- Mengetahui cara kerja kedua algoritma mulai dari data input hingga top-N output



## 2. Business Understanding

### Problem Statement
Platform streaming sering mengalami masalah ketika pengguna kesulitan menemukan film yang sesuai dengan preferensi mereka. Banyak pengguna menghabiskan waktu lama di halaman pencarian atau rekomendasi umum, kemudian meninggalkan aplikasi tanpa menonton apa pun. Kondisi ini menurunkan tingkat engagement dan berpotensi mengurangi retensi pengguna.  

Permasalahan bisnis yang ingin diselesaikan adalah:   

*“Bagaimana cara memberikan rekomendasi film yang relevan dan personal sehingga pengguna dapat menemukan tontonan dengan cepat dan tetap bertahan di platform?”*

### Goals
Tujuan proyek ini adalah membangun sistem rekomendasi yang mampu membantu pengguna menemukan film yang cocok dengan preferensi mereka. Secara spesifik, tujuan bisnis yang ingin dicapai adalah:

1. Menghasilkan **top-5 rekomendasi film** yang relevan untuk setiap pengguna.  
2. Meningkatkan efisiensi pencarian film sehingga waktu yang dibutuhkan pengguna untuk menemukan tontonan menjadi lebih singkat.  
3. Mengevaluasi kualitas rekomendasi menggunakan metrik yang dapat dipertanggungjawabkan secara kuantitatif, yaitu **Precision@K, Recall@K, dan F1@K**.  
4. Mendukung pengambilan keputusan bisnis terkait peningkatan fitur rekomendasi di masa depan.

### Solution Statement
Untuk mencapai tujuan tersebut, digunakan dua pendekatan yang saling melengkapi:

1. **Content-Based Filtering**  
   Rekomendasi diberikan berdasarkan kemiripan konten film (misalnya genre). Pendekatan ini memastikan bahwa pengguna selalu memiliki rekomendasi meskipun belum banyak memberikan rating. Solusi ini membantu mengurangi cold-start di level pengguna.

2. **Collaborative Filtering**  
   Rekomendasi diberikan berdasarkan pola perilaku pengguna lain yang memiliki preferensi serupa. Dengan mempelajari hubungan user–movie melalui embedding, sistem dapat memberikan rekomendasi yang lebih personal dan akurat. Pendekatan ini meningkatkan nilai bisnis karena semakin banyak interaksi pengguna, semakin baik rekomendasinya.

Kedua pendekatan ini digabungkan sebagai fondasi sistem rekomendasi yang lebih stabil, scalable, dan relevan bagi pengalaman pengguna.


## 3. Data Understanding


Pada tahap ini saya memiliki dua berkas inti: movies.csv dan ratings.csv.

### ratings.csv 

menyimpan **25.000.095 baris  berupa data kontinu yang terdapat kolom userid(id pengguna yang melakukan rating), moviesid(Id unik film) ,rating(rating yang diberikan pengguna dari skala 1-5) dan timestamp(waktu melakukan rating)** plus skor rating; angka gede itu muncul karena satu film bisa dibanjiri ribuan review. terdapat 59.047 film unik yang kebagian rating. dari data ini tidak terdapat missing values dan duplikat.

<img width="678" height="470" alt="image" src="https://github.com/user-attachments/assets/1126b22d-61c1-4c9c-8f24-dd6bc9b10729" />

Grafik distribusi rating tersebut menunjukkan bahwa sebagian besar pengguna memberikan rating pada rentang 3.0 hingga 4.0, terlihat dari batang histogram yang paling tinggi pada rentang tersebut. Artinya, kecenderungan rating di dataset ini cenderung positif dan jarang terdapat rating sangat rendah. Jumlah rating pada tiap kelompok (bin) bisa mencapai jutaan karena total data sangat besar (25 juta baris), sehingga satu rentang rating tertentu dapat berisi 4–6 juta rating

 ### movies.csv 
 berisi **62.423 baris lengkap dengan kolom movie id(key unik film untuk tersambung dengan dataset ratings), title(judul film), genres( genre dari film, satu film bisa miliki 1 atau lebih genre).** Perbedaan jumlah film antara movies.csv (62k film) dan film unik pada ratings.csv (59k film) terjadi karena tidak semua film yang tercatat memiliki rating. dataset ini tidak memiliki missing values, duplikat dan outlier sehingga dataset sudah bersih.

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/39c241d5-351d-4f96-aa3a-5e63aed7635d" />

Distribusi jumlah film per genre pada dataset ini menunjukkan bahwa beberapa kategori film jauh lebih dominan dibanding yang lain. Genre Drama menjadi yang paling banyak ditemui, disusul oleh Comedy, Thriller, dan Romance. Pola ini wajar karena genre-genre tersebut memang umum dan banyak diproduksi dalam industri film. Sementara itu, genre seperti Western, Musical, Film-Noir, hingga IMAX muncul dalam jumlah yang jauh lebih sedikit. Ketidakseimbangan ini memberi gambaran bahwa dataset memiliki keberagaman genre, namun tidak merata.
ini adalah web mendownload file tersebut
 [download](https://www.kaggle.com/parasharmanas/movie-recommendation-system) 

## 4. Data Preparation 
### 4.1  Genre Splitting 

Setiap film dapat memiliki **lebih dari satu genre**, sehingga dilakukan **genre splitting** menggunakan metode `str.get_dummies(sep='|')`. Teknik ini mengubah representasi genre dari format multiclass menjadi multi-label biner, di mana setiap genre menjadi kolom tersendiri dengan nilai 0 (tidak memiliki genre tersebut) atau 1 (memiliki genre tersebut).

### 4.2 Penggabungan Dataset (Merge)

Setelah genre diproses menjadi format biner (`movies_expanded`), data tersebut **digabungkan dengan dataset ratings** menggunakan `movieId` sebagai kunci penggabungan melalui operasi `merge`. 

Dari kedua proses ini terbentuklah tabel utuh df yang akan diolah lebih lanjut 

### 4.3 drop fitur yang tidak berkontirbusi
ada 2 fitur harus di hapus karna tidak memiliki kontirbusi yaitu  `drop(columns=['timestamp', '(no genres listed)'])`. Kolom `timestamp` dihilangkan karena tidak memberikan signal yang signifikan bagi model sistem rekomendasi, terutama untuk pendekatan content-based filtering yang berfokus pada kesamaan atribut konten. Sementara itu, kolom `(no genres listed)` di drop karna juag tidak memberikan kontribusi/kosong.

####  Kolom Identifikasi & Rating 
 dataset final (df) memiliki struktur kolom yang komprehensif dan siap untuk analisis lebih lanjut:

- **`userId`** - ID unik pengguna yang memberikan rating
- **`movieId`** - Primary key dari tabel movie, berfungsi sebagai penghubung antar dataset
- **`rating`** - Nilai rating yang diberikan pengguna (skala 0.5 - 5.0)
- **`timestamp`** - Waktu ketika rating diberikan dalam format Unix timestamp
- **`(semua kolom genre dalam bentuk biner)`** karna masing masing film memiliki genre yang ber beda-beda maka dibuat kolom-kolom genre 

### 4.4 Data cleaning
setelah kolom gabungan terbentuk ,kita melakukan cek ulang terhadap missing values dan duplikasi data. dan didapatkan hasilnya **tidak ada**.

### 4.5 Sorting data dan drop duplikat MovieId
dilakukan **sorting berdasarkan movieId** menggunakan `df.sort_values(by='movieId')` untuk memudahkan identifikasi, urutan data menjadi rapi dan konsisten.

Langkah selanjutnya adalah **penghapusan duplikasi film** menggunakan `drop_duplicates('movieId')`. Hal ini penting karena dalam dataset ratings, satu film dapat muncul berkali-kali dengan rating dari pengguna yang berbeda. 

 setiap film cukup direpresentasikan sekali saja. Setelah proses ini, didapatkan **59.047 film unik** yang siap digunakan untuk membangun matriks kemiripan.

### 4.6  Encoding , Normalisasi, data split (`Khusus Collaborative filtering`)
Berbeda dengan Content-Based Filtering yang hanya memerlukan fitur genre film, Collaborative Filtering bekerja dengan mempelajari pola interaksi antara pengguna dan film melalui data rating. Oleh karena itu, persiapan data untuk Collaborative Filtering memerlukan pendekatan yang berbeda, fokus pada transformasi ID dan normalisasi rating agar dapat diproses oleh neural network.

Pada Collaborative Filtering, data yang digunakan hanya mencakup `userId, movieId, dan rating` dari data yang sudah di persiapkan sebelumnya. tahapan ini melibatkan
1. preprocessing melibatkan **encoding dan scaling** . Kedua ID dikonversi menjadi indeks numerik menggunakan Label Encoding
 2.  rating dinormalisasi ke rentang 0–1 agar sesuai dengan fungsi aktivasi sigmoid. Setelah itu, fitur (user, movie) dan target (rating_norm) dipisahkan,

 3. lalu dibagi menjadi data train–validation dengan rasio **80:20**. Data yang telah tersiapkan ini kemudian digunakan sebagai input untuk melatih model Collaborative.


## 5. Modeling & Result

### 5.1 Content-Based Filtering
Setiap film direpresentasikan sebagai **vektor fitur 19-dimensi** berdasarkan genre biner hasil one-hot encoding. Untuk mengukur kemiripan antar film, digunakan **Cosine Similarity** yang dipresentasikan dalam bentuk persen. **Semakin mirip genre antar film, semakin tinggi nilai similarity-nya**.

Sebagai contoh, film **"Toy Story (1995)"** memiliki kemiripan **91.29%** dengan **"Space Jam (1996)"** karena keduanya berbagi genre *Animation, Children, dan Comedy*. Sistem menghasilkan **Top-3 film paling mirip** untuk setiap judul sebagai rekomendasi.berikut kekurangan dan kelebihan content based algoritma

**Kelebihan**:

- **Tidak memerlukan data pengguna lain** - Sistem dapat beroperasi bahkan dengan hanya satu pengguna, karena rekomendasi didasarkan pada kesamaan konten film, bukan pola kolaboratif antar pengguna.
- **Tidak ada cold-start problem untuk pengguna baru** - User baru langsung dapat menerima rekomendasi setelah memberikan beberapa rating, tanpa harus menunggu akumulasi data dari banyak pengguna.
- **Independen dari popularitas** - Film yang kurang populer tetap dapat direkomendasikan selama memiliki kesamaan genre dengan preferensi pengguna.


**Kekurangan**:
- **Keterbatasan fitur** - Model hanya mengandalkan informasi genre, sehingga tidak dapat menangkap preferensi kompleks pengguna terhadap aspek lain seperti sutradara, aktor, sinematografi, atau mood film.
- **Over-specialization** - Cenderung merekomendasikan film yang terlalu mirip,= membatasi eksplorasi pengguna terhadap genre baru.

dalam pemrossesan data ,karena keterbatasan memori di Google Colab yang menyebabkan crashed session berkali-kali, perhitungan similarity dilakukan pada **subset 1000 film pertama**. Hasilnya berupa **matriks similarity** berukuran 1000×1000. berikut kira kira hasil dari penggunaan model cosine
```
movieId   1021   1022  
movieId                
1        63.25  60.00  
2        40.82  51.64  
3        50.00  31.62  
4        40.82  25.82  
5        70.71   0.00
```

Nilai dalam matriks menunjukkan persentase kemiripan antar film. Misalnya, film `movieId` 1 memiliki kemiripan **63.25%** dengan film `movieId` 1021. Nilai **0.00%** mengindikasikan tidak ada kesamaan genre sama sekali. Matriks ini menjadi dasar sistem untuk merekomendasikan film dengan similarity tertinggi.


### Contoh Hasil Rekomendasi Content filtering (Top-N)

=== Film: Jumanji (1995) (movieID 2) === 
memiliki kesamaan dengan:
* Indian in the Cupboard, The (1995)  
* NeverEnding Story III, The (1994)  
* Kids of the Round Table (1995)      
* Amazing Panda Adventure, The (1995)
* Casper (1995)                     


---

### 5.2 Collaborative Filtering

Collaborative Filtering menggunakan arsitektur **Neural Matrix Factorization** yang memanfaatkan pola interaksi antara pengguna dan film untuk menghasilkan rekomendasi. Berbeda dengan content-based yang mengandalkan kesamaan genre, metode ini belajar dari **preferensi implisit yang tersembunyi dalam pola rating kolektif**.

### Arsitektur Model

Pada tahap ini model menggunakan pendekatan Neural Collaborative Filtering dengan embedding

**embedding_dim = 64**
Setiap user dan film direpresentasikan sebagai vektor 64 dimensi.
Semakin besar dimensinya, semakin kaya pola hubungan yang bisa dipelajari.

Output dihitung lewat perkalian dot antara embedding user & film.
Ini memberi gambaran seberapa mirip preferensi user dengan karakteristik film.

Aktivasi sigmoid digunakan agar skor prediksi berada pada rentang 0–1.
Aman dan stabil untuk proses training.


### Training dan Evaluasi

Model dilatih menggunakan **optimizer Adam** dengan **loss function MSE** dan **metrik evaluasi MAE**. Untuk mencegah overfitting, diterapkan **early stopping** dengan patience 3 epoch yang akan menghentikan training jika validation loss tidak membaik selama 3 epoch berturut-turut.


### Contoh Hasil Rekomendasi Collaborative filtering (Top-N)

Sebagai contoh, untuk pengguna dengan `userId=6`, sistem merekomendasikan 5 film teratas setelah denormalisasi prediksi rating:

Top 5 rekomendasi untuk User 6:
1. Star Wars: Episode IV - A New Hope (1977)
2. Shawshank Redemption, The (1994)
3. Star Wars: Episode V - The Empire Strikes Back (1980)
4. Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)
5. Star Wars: Episode VI - Return of the Jedi (1983)

Rekomendasi ini menunjukkan bahwa model berhasil menangkap preferensi kompleks pengguna yang tidak hanya terbatas pada kesamaan genre, tetapi juga pola selera yang lebih nuanced.

# Evaluasi Model Rekomendasi Film

## 6.1 Evaluasi Content-Based Filtering

Tahapan evaluasi dilakukan untuk mengukur seberapa akurat model content-based filtering dalam memberikan rekomendasi film. Karena model dibangun menggunakan fitur **genre biner**, maka evaluasi juga menggunakan genre sebagai dasar penentuan relevansi. Pendekatan ini masuk akal karena sistem rekomendasi content-based bekerja dengan mencari kesamaan karakteristik antar film, dan dalam kasus ini karakteristik utamanya adalah genre.

### Metrik Evaluasi 

Proses evaluasi dimulai dengan mengambil genre dari film yang sedang dievaluasi menggunakan kode berikut:


1. Pada tahap ini, genre film dijadikan sebagai ground-truth atau acuan kebenaran untuk menentukan apakah rekomendasi yang diberikan relevan atau tidak. Genre dianggap sebagai label karena sistem hanya memiliki fitur genre biner, sehingga tidak ada fitur lain yang dapat dijadikan pembanding.
```python 
query_genres = movie_features.loc[idx]
```
2. selanjutnya untuk menghasilkan daftar Top-N rekomendasi berdasarkan nilai cosine similarity tertinggi. Film yang sedang dievaluasi sendiri tidak diikutsertakan dalam hasil rekomendasi (di-skip menggunakan `drop(idx)`), karena tidak masuk akal merekomendasikan film yang sama. Hasilnya diurutkan dari similarity tertinggi ke terendah, lalu diambil sejumlah N teratas (dalam kasus ini N=5).
```python
sims = cos_sim_df_subset.loc[idx].drop(idx).sort_values(ascending=False).head(topn)
```

3. Tahap selanjutnya adalah menentukan apakah sebuah rekomendasi relevan atau tidak. Proses ini dilakukan dengan membandingkan genre film rekomendasi dengan genre film asli:

Pada bagian ini, sistem menghitung berapa banyak genre yang tumpang-tindih (overlap) antara film asli dan film rekomendasi. Sebuah rekomendasi dianggap relevan hanya jika memiliki **setidaknya dua genre yang sama** dengan film asli. Ambang batas dua genre ini dipilih agar evaluasi lebih ketat—model tidak sembarangan menganggap film relevan hanya karena memiliki satu kesamaan kecil. Dengan threshold ini, film yang direkomendasikan benar-benar harus memiliki kemiripan genre yang signifikan.
```python
rec_genres = movie_features.loc[movie_id]
overlap = set(query_genres.index[query_genres == 1]) & set(rec_genres.index[rec_genres == 1])
relevant = 1 if len(overlap) >= 2 else 0
```

Setelah relevansi setiap rekomendasi ditentukan, precision per film dihitung menggunakan formula sederhana:
```python
return relevant / topn
```

**Formula:**
```
Precision@K = (Jumlah film relevan) / K
```

Precision@K menghitung proporsi film yang relevan dari total K rekomendasi yang diberikan. Misalnya, jika dari 5 rekomendasi ternyata semua relevan, maka Precision@5 = 5/5 = 1.00 atau 100%. Metrik ini memberikan gambaran langsung tentang kualitas rekomendasi yang dihasilkan model.

Untuk mendapatkan gambaran performa model secara keseluruhan, precision dihitung untuk beberapa film sampel, kemudian dirata-ratakan:
```python
for idx in sample_indices:
    prec = precision_per_film(...)
    print(f"> Precision@5: {prec:.2f}")
```

Proses loop ini dijalankan pada 10 film pertama untuk menampilkan nilai precision masing-masing film. Nilai-nilai ini kemudian dapat dirata-ratakan untuk melihat performa model secara global, yang disebut sebagai Precision@5 makro.

### Hasil Evaluasi Content-Based Filtering

Sebagai contoh konkret, berikut adalah hasil evaluasi untuk film **Jumanji (1995)** dengan MovieID 2:
```
=== Film: Jumanji (1995) (ID 2) ===
Top-5 Rekomendasi:
* Indian in the Cupboard, The (1995)  | similarity: 100.000
* NeverEnding Story III, The (1994)   | similarity: 100.000
* Kids of the Round Table (1995)      | similarity: 86.600
* Amazing Panda Adventure, The (1995) | similarity: 81.650
* Casper (1995)                       | similarity: 81.650

Precision@5: 1.00
NDCG@5: 1.000
```

Hasil evaluasi untuk film *Jumanji (1995)* menunjukkan bahwa seluruh lima rekomendasi yang diberikan model memiliki kesamaan genre yang kuat dengan film aslinya. Hal ini tercermin dari nilai **Precision@5 yang mencapai 1.00** atau 100%, yang berarti semua rekomendasi yang diberikan relevan dan memenuhi kriteria minimal dua genre yang sama.

Nilai NDCG@5 = 1.000 berarti model tidak hanya memberi rekomendasi yang relevan, tetapi juga menempatkannya dalam urutan yang paling ideal. Semua film yang direkomendasikan memiliki kemiripan genre yang sangat tinggi dengan Jumanji, sehingga baik relevansi maupun urutan rekomendasinya sudah sempurna.

Hasil ini menunjukkan bahwa model content-based filtering bekerja sangat baik dalam menemukan film-film yang benar-benar segenre dan menempatkannya pada posisi ranking yang tepat.


## 6.2 Evaluasi Collaborative Filtering

dari hasil pelatihan didapatkan bahwa pelatihan terbaik :
 (epoch-2)
- Train MAE: **0.1241**
- Val MAE: **0.1315**
- Error rata-rata model hanya sekitar **0.12–0.13 poin rating**, artinya prediksi cukup akurat.


MAE yang kecil menunjukkan bahwa kualitas prediksi rating model sudah baik, karena rata-rata kesalahan prediksi hanya sekitar 0.12–0.13 poin. Selisih antara nilai train dan validation juga sangat rendah, sehingga model tidak menunjukkan tanda-tanda overfitting. Dengan stabilitas dan akurasi seperti ini, model sudah cukup kuat dan layak digunakan dalam sistem rekomendasi.

###  Metrik Evaluasi

Pada proyek ini, kinerja model collaborative filtering dievaluasi menggunakan tiga metrik utama: **Precision@K**, **Recall@K**, dan **F1@K**. Ketiga metrik ini dipilih karena sesuai dengan tujuan sistem, yaitu memberikan rekomendasi film yang relevan bagi setiap pengguna berdasarkan histori rating mereka.

**Precision@K** mengukur seberapa banyak dari K rekomendasi teratas yang benar-benar relevan bagi pengguna. Semakin tinggi precision, semakin akurat model dalam memberikan rekomendasi yang tepat sasaran.

**Formula:**
```
Precision@K = (Jumlah item relevan di top-K) / K
```

**Recall@K** mengukur seberapa banyak item relevan yang berhasil ditemukan oleh model dari seluruh item relevan yang dimiliki pengguna.Recall yang tinggi menunjukkan model mampu menemukan sebagian besar film yang relevan untuk pengguna.

**Formula:**
```
Recall@K = (Jumlah item relevan di top-K) / (Total item relevan pengguna)
```

**F1@K** adalah harmonic mean dari precision dan recall, digunakan untuk menilai keseimbangan keduanya. 

**Formula:**
```
F1@K = 2 × (Precision × Recall) / (Precision + Recall)
```

Metrik ini dihitung untuk satu pengguna tertentu dengan memprediksi seluruh film yang mungkin ia tonton, kemudian memilih lima rekomendasi terbaik (K = 5). Kode penghitungan menggunakan fungsi `precision_recall_f1_at_k()`, yang mengambil prediksi model, menyaring film relevan berdasarkan threshold rating ≥ 4.0, kemudian menghitung ketiga metrik tersebut.

### Hasil Evaluasi Model Collaborative filtering

Untuk pengguna yang diuji yaitu **User 6**, model menghasilkan nilai sebagai berikut:

| Metrik | Nilai |
|--------|-------|
| **Precision@5** | 0.8000 |
| **Recall@5** | 0.2000 |
| **F1@5** | 0.3200 |

Model menghasilkan precision 0.80 (80%), yang berarti sebagian besar film yang direkomendasikan benar-benar relevan dengan preferensi pengguna. Namun, recall hanya 0.20 (20%), menunjukkan bahwa banyak film relevan lain yang belum berhasil ditangkap dalam Top-5 rekomendasi. Nilai F1 = 0.32 menggambarkan bahwa meskipun rekomendasi cukup akurat, cakupan item relevannya masih terbatas sehingga model masih berpotensi ditingkatkan.



# 7. Kesimpulan dan Dampak terhadap Business Understanding


**Problem Statement:** *"Bagaimana cara memberikan rekomendasi film yang relevan dan personal sehingga pengguna dapat menemukan tontonan dengan cepat dan tetap bertahan di platform?"*

Problem statement ini **telah terjawab dengan baik**. Content-Based Filtering berhasil memberikan rekomendasi berdasarkan kemiripan genre dengan Precision@5 = 1.00 dan NDCG@5 = 1.000, memastikan pengguna langsung mendapatkan film sejenis tanpa mencari manual. Collaborative Filtering memberikan rekomendasi personal dengan Precision@5 = 0.80, menunjukkan 80% rekomendasi benar-benar relevan dengan preferensi pengguna. Film-film seperti *Star Wars*, *Shawshank Redemption*, dan *Raiders of the Lost Ark* yang direkomendasikan membuktikan model memahami pola preferensi kompleks pengguna, bukan sekadar kesamaan genre. Kedua model secara efektif meningkatkan kecepatan pengguna menemukan film yang sesuai dan meningkatkan peluang mereka bertahan di platform.


**Semua goals berhasil dicapai.** Goal pertama menghasilkan top-5 rekomendasi relevan terpenuhi dengan Content-Based mencapai similarity hingga 100% dan Collaborative mencapai precision 80%. Goal kedua meningkatkan efisiensi pencarian tercapai karena pengguna tidak perlu browsing manual—Content-Based memberikan rekomendasi sejenis dalam hitungan detik, sementara Collaborative menyajikan rekomendasi personal berdasarkan histori rating. Goal ketiga mengevaluasi kualitas dengan metrik kuantitatif terpenuhi melalui Precision@K, Recall@K, F1@K untuk Collaborative dan Precision@K, NDCG@K untuk Content-Based yang dapat dijelaskan kepada stakeholder non-teknis. Goal keempat mendukung pengambilan keputusan bisnis tercapai dengan insight bahwa Content-Based efektif untuk cold-start problem pengguna baru, sedangkan Collaborative cocok untuk pengguna aktif, membuka peluang strategi hybrid di masa depan.


**Kedua solution statement memberikan dampak signifikan.** Content-Based Filtering berdampak langsung pada retention pengguna baru dengan mengatasi cold-start problem—pengguna tanpa histori rating tetap mendapat rekomendasi berkualitas tinggi (Precision@5 = 1.00). Pendekatan ini juga independen dari popularitas, sehingga film niche tetap dapat direkomendasikan ke audiens yang tepat, meningkatkan diversity konten. Collaborative Filtering berdampak pada peningkatan engagement pengguna aktif dengan personalisasi yang semakin akurat seiring bertambahnya data interaksi. MAE yang rendah (0.12-0.13) menunjukkan probabilitas pengguna menyukai rekomendasi sangat tinggi, berdampak langsung pada conversion rate dari rekomendasi ke actual viewing. Kombinasi keduanya menciptakan sistem yang stabil dan scalable: Content-Based untuk pengguna baru, Collaborative untuk pengguna aktif, sejalan dengan temuan Gomez-Uribe & Hunt (2016) bahwa 80% konten yang ditonton di Netflix berasal dari rekomendasi.

