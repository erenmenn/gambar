# Laporan Proyek Machine Learning  
**Judul Proyek**: *"Prediksi Kemurnian dan Harga Madu Menggunakan Machine Learning"*  
**Nama**: Emirsyah Rafsanjani  
**Dataset**: [Kaggle - Predict Purity and Price of Honey](https://www.kaggle.com/stealthtechnologies/predict-purity-and-price-of-honey)



##  Domain Proyek

### 1. Latar Belakang
Madu merupakan salah satu produk alami yang memiliki banyak manfaat bagi kesehatan. Namun, tidak semua madu yang beredar di pasaran benar-benar murni. Banyak produk madu yang telah dicampur dengan gula, sirup, atau bahan lainnya untuk menekan biaya produksi. Hal ini tentunya merugikan konsumen, terutama jika mereka membeli madu dengan harga mahal tetapi kualitasnya rendah.

Dengan memanfaatkan teknologi *machine learning*, kita dapat membangun model yang mampu memprediksi kemurnian madu dan menaksir harga yang sesuai berdasarkan karakteristik fisik dan kimia yang dapat diukur. Proyek ini bertujuan untuk membantu konsumen dan produsen dalam menentukan kualitas dan harga madu secara objektif.

### 2. Mengapa Masalah Ini Penting?
- Melindungi konsumen dari madu palsu.  
- Membantu produsen dalam menentukan harga jual.  
- Meningkatkan transparansi dan kepercayaan dalam pasar madu.

### 3. Referensi
Dataset yang digunakan dalam penelitian ini berasal dari platform Kaggle, yaitu “Predict Purity and Price of Honey” yang dirilis oleh Stealth Technologies. Kaggle merupakan platform kredibel milik Google yang digunakan secara luas dalam penelitian akademik dan industri untuk pemodelan data. Stealth Technologies sebagai penyusun dataset memiliki rekam jejak yang kuat dan mendapatkan berbagai penghargaan (medals) dalam kategori Datasets, Notebooks, dan Competitions, sehingga meningkatkan kredibilitas dataset.
- Kaggle Dataset: *Predict Purity and Price of Honey* oleh Stealth Technologies  
  [https://www.kaggle.com/stealthtechnologies/predict-purity-and-price-of-honey](https://www.kaggle.com/stealthtechnologies/predict-purity-and-price-of-honey)
- Scikit-learn Documentation  
  [https://scikit-learn.org/](https://scikit-learn.org/)



##  Business Understanding

###  Problem Statements
Bagaimana cara memprediksi **harga madu** berdasarkan karakteristiknya?

###  Goals
1. Membangun model regresi untuk memprediksi harga madu

2. Menentukan algoritma regresi terbaik untuk prediksi harga madu


## Solution Statements

Untuk mencapai tujuan tersebut, saya menggunakan pendekatan berikut:

- Membuat model dasar menggunakan algoritma sederhana seperti Linear Regression sebagai pembanding awal. Baseline ini bertujuan melihat sejauh mana model sederhana mampu memprediksi harga madu.  

- Menjalankan dan membandingkan beberapa algoritma regresi

- Evaluasi model menggunakan metrik yang sesuai untuk memilih model terbaik



## Data Understanding

### Sumber Data
Dataset diambil dari kaggke https://www.kaggle.com/stealthtechnologies/predict-purity-and-price-of-honey) dengan **247.903 baris data** dan **11 kolom**.

###  Fitur-fitur:

- vCS – Skor warna madu (rentang 1–10).

- Density – Tingkat kepadatan madu dalam satuan g/cm³.

- WC – Persentase kadar air yang terkandung dalam madu.

- pH – Tingkat keasaman madu.

- EC – Konduktivitas listrik madu yang diukur dalam mS/cm.

- F – Persentase kandungan fruktosa.

- G – Persentase kandungan glukosa.

 - Pollen_analysis – Jenis bunga asal madu (bersifat kategorikal).

- viscosity – Tingkat kekentalan madu dalam satuan cP (centipoise).

- purity – Variabel target untuk klasifikasi, dengan nilai antara 0.61 hingga 1.00.

- Price – Variabel target untuk regresi, menyatakan harga madu 

### Eksplorasi Data
- Tidak ada data hilang.  
- Tidak ada outlier berdasarkan IQR method.  
- Visualisasi distribusi fitur dilakukan dengan histogram dan boxplot.  
- Korelasi antar fitur dianalisis dengan heatmap.


##  Data Preparation

### Transformasi Harga (Log Transformation)
Harga madu memiliki distribusi **right-skewed** ,jadi transformasi log mengubah bentuk mendekati **normal**.

###  Splitting Data
Dataset dibagi menjadi train set dan test set menggunakan train_test_split(). 80 % train : 20 % test

X = Semua fitur --- tanpa 'Price' dan 'Log_price'

y = log_price --- Target label

jika price diikutkan kedalam train ,akan terjadi leakage atau 'menyontek data' karna price adalah bentuk awal dari log_price sehingga perlu di drop 

###  scaling dan encoding
- Fitur numerik diskalakan menggunakan StandardScaler agar berada pada skala yang sama.
- Fitur Pollen_analysis bersifat kategorikal sehingga perlu diubah menjadi numerik.
Digunakan teknik One-Hot Encoding agar setiap kategori direpresentasikan sebagai kolom biner. Memakai One hot karna kolom bersifat non sequential
 ## Modeling
 Saya menggunakan 3 algoritma untuk dilihat perbedaan ketiganya:  

###  KNN Regressor  
KNN sederhana dan tidak memerlukan asumsi distribusi data. Dengan 7 tetangga terdekat, model tetap dapat menangkap pola non linear



### Random Forest Regressor  
Random Forest kuat terhadap overfitting, bisa handle fitur berjumlah banyak, dan memberikan feature importance. Kombinasi 300 pohon dengan kedalaman 10 cukup dalam untuk menangkap pola tanpa membuat model terlalu kompleks.

###  AdaBoost Regressor    
AdaBoost meningkatkan performa dengan fokus pada error pohon sebelumnya. Learning rate kecil (0.08) dan 250 iterasi memastikan peningkatan bertahap tanpa overfitting. 

##  Evaluation

###  K-Nearest Neighbors (KNN)
Model KNN menunjukkan performa yang cukup baik dengan **MSE 0.000421** pada training dan **0.000575** pada testing. RMSE yang dihasilkan adalah sekitar **0.0205** dan **0.0240**, sedangkan nilai **R² mencapai 0.9983** pada training dan **0.9976** pada testing. Hal ini menunjukkan bahwa KNN mampu memprediksi harga madu dengan akurat tinggi, meskipun terdapat sedikit peningkatan error di data testing yang masih  normal.

###  Random Forest  
Random Forest menghasilkan **MSE 0.001912** pada training dan **0.001922** pada testing, dengan **RMSE sekitar 0.0437** dan **0.0438**. Nilai **R² nya adalah 0.9922** untuk training dan **0.9921** untuk testing. Meskipun nilai R² masih sangat tinggi, error yang lebih besar dibandingkan dua model lainnya menunjukkan bahwa Random Forest kurang optimal dalam menangkap pola harga madu pada dataset ini.

### AdaBoost  
adaBoost memberikan hasil terbaik dengan **MSE sangat kecil: 0.000082** pada training dan **0.000081** pada testing. **RMSE nya hanya sekitar 0.0093** dan **0.0092**, serta nilai **R² hampir sempurna: 0.9997** baik pada training maupun testing.  
Ini menunjukkan bahwa **AdaBoost adalah model paling akurat, stabil, dan mampu generalisasi dengan sangat baik** dalam memprediksi harga madu pada dataset yang digunakan.

#### Hasil diatas adalah evaluasi dari skala log price dengan ketiga bentuk model ,berikut adalah Evaluasi Hasil Menggunakan **Harga Asli** Setelah Inverse Log Transform serta Perbandingan **Linear Regression** dengan Model Lain

### Confidence Interval RMSE (Skala Log)
| Model           | Lower Bound | Upper Bound |
|----------------|-------------|-------------|
| KNN            | 0.0766      | 0.0804      |
| RandomForest   | 0.1238      | 0.1248      |
| Boosting       | 0.1532      | 0.1544      |
| LinearReg      | 0.0106      | 0.0107      |

### MAPE (Skala Harga Asli)
| Model           | MAPE (%)    |
|----------------|-------------|
| KNN            | 3.62        |
| RandomForest   | 11.18       |
| Boosting       | 12.31       |
| LinearReg      | 0.93        |

### RMSE (Skala Harga Asli)
| Model           | RMSE (Harga Asli) |
|----------------|-------------------|
| KNN            | 35.00            |
| RandomForest   | 74.75            |
| Boosting       | 103.18           |
| LinearReg      | 6.69             |

 Meskipun ketiga model non-linear menghasilkan prediksi yang tampak dekat pada skala log, selisih kecil pada ruang log ternyata menghasilkan perbedaan harga yang sangat besar ketika dikembalikan ke skala aslinya. Hal ini tercermin dari nilai MAPE dan RMSE yang relatif tinggi, di mana error harga dapat mencapai 35 hingga lebih dari 100 juta rupiah.

Linear Regression Memberikan performa yang jauh lebih baik dan paling stabil di antara seluruh model. Model ini menghasilkan interval kepercayaan RMSE yang sangat sempit pada skala log, menandakan ketidakpastian prediksi yang rendah. dan setelah dicoba melakukan testing menggunakan linear regression kepada harga asli hasilnya terlihat sangat baik dimana Actual Price 615.18  diprediksi menjadi 618.962293



##  Kesimpulan

dari serangkaian percobaan dan evaluasi yang telah dilakukan, dapat disimpulkan bahwa:

Pada skala log price, ketiga model non-linear (KNN, Random Forest, dan AdaBoost) menunjukkan performa yang sangat baik dengan nilai MSE dan RMSE yang kecil serta R² yang tinggi. 

Namun, ketika hasil prediksi dikembalikan ke skala harga asli, perbedaan kecil pada ruang log berubah menjadi selisih harga yang besar. Hal ini menyebabkan model non-linear menghasilkan MAPE dan RMSE yang cukup tinggi, dengan error prediksi yang cukup besar.

Interval kepercayaan RMSE pada skala log sangat sempit, menandakan rendahnya ketidakpastian prediksi.

Perbedaan performa ini menunjukkan bahwa dataset memiliki pola hubungan yang cenderung linear, sehingga model Linear Regression dapat menangkap struktur datanya dengan lebih baik dibandingkan model non-linear.

Secara keseluruhan, Linear Regression merupakan model terbaik untuk memprediksi harga madu pada dataset ini, terutama setelah dilakukan inverse transform ke skala harga asli. Model ini tidak hanya paling akurat, tetapi juga paling stabil dan mudah diinterpretasikan.

##  Referensi

1. Kaggle Dataset - *Predict Purity and Price of Honey*  
   [https://www.kaggle.com/stealthtechnologies/predict-purity-and-price-of-honey](https://www.kaggle.com/stealthtechnologies/predict-purity-and-price-of-honey)

2. Scikit-learn Documentation  
   [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

