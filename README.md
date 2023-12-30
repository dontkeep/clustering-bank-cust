# Laporan Proyek Machine Learning

### Nama : Muchammad Romadona

### Nim : 211351085

### Kelas : Malam B

## Domain Proyek
Project web app ini digunakan untuk melakukan clustering pada dataset Customer Credit Card. Bisa berfungsi untuk menentukan pelanggan mana yang masuk pada kelompok/member tertentu.

## Business Understanding
Web ini bertujuan untuk mengelompokkan pelanggan-pelanggan bank agar bisa dengan mudah memahami kebutuhan dan preferensi pelanggan dengan lebih baik.

### Problem Statement
Sulitnya untuk memberikan apa yang diinginkan oleh pelanggan-pelanggan yang berbeda.

### Goals
Memudahkan pihak bank dalam memahami kebutuhan pelanggannya dan bisa memberikan preferensi yang sesuai bagi semua pelanggan-pelanggannya.

### Solution Statements
- Membuatkan web pengelompokan pelanggan untuk mengetahui dan mengenal pelanggan dengan lebih mudah.

## Data Understanding
Dataset ini saya dapatkan dari website Kaggle. Ianya berisikan data-data pelanggan sebuah bank dengan jumlah 7 kolom dan 660 baris data. Dataset ini sudah cukup bersih untuk digunakan pengelompokan.
[Credit Card Custom](https://www.kaggle.com/datasets/aryashah2k/credit-card-customer-data)

### Variabel-variabel pada Diabetes Prediction adalah sebagai berikut:
- SI_No : Merepresantasikan Serial id number pelanggan. [int, 1-660]
- Customer Key : Merepresantasikan kunci unik pelanggan. [int, 11,300-99,800]
- Avg_Credit_Limit : Merepresantasikan limit credit rata-rata pelanggan. [int, 3000-200,000]
- Total_Credit_Cards : Merepresantasikan jumlah kartu credit yang dimiliki pelanggan. [int, 1-10]
- Total_visits_bank : Merepresantasikan jumlah berapa kali dilakukan pengunjungan oleh pelanggan ke bank. [int, 0-5]
- Total_visits_online : Merepresantasikan jumlah berapa kali dilakukan pengunjungan oleh pelanggan dalam jaringan. [int, 0-15]
- Total_calls_made : Merepresantasikan jumlah panggilan yang dibuat oleh pelanggan. [int, 0-10]

## Data Preparation
Untuk bagian data preparation ini saya akan melakukan proses EDA dan pre-processing yaaa!!

### Import Dataset
```bash
from google.colab import files
files.upload()
```
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
```bash
!kaggle datasets download -d aryashah2k/credit-card-customer-data
```
```bash
!unzip credit-card-customer-data.zip -d dataset
!ls dataset
```
Selesai mengunduh datasets yang dipilih, selanjutnya saya akan mengimport library yang dibutuhkan serta mengextract datasets yang tadi diunduh.
### Import All Library 
```bash
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
```
### Data Discovery
```bash
df = pd.read_csv('dataset/Credit Card Customer Data.csv')
df.head()
```
Datasets telah berhasil dibaca menggunakan pandas, terdapat 7 columns pada datasetsnya namun yang akan gunakan nanti hanyalah 2 columns, yaitu column Avg_Credit_Limit dan Total_Credit_Cards. Sebelumnya saya akan melihat apakah datasets ini memiliki nilai null atau tidak.
```bash
df.isnull().sum()
```
Tidak ada yang null ya, selanjutnya kita akan melihat korelasi antar columns menggunakan heatmap.
```bash
df.info()
```
Semua kolomnya berjenis integer ya! ini memudahkan kita dalam proses preprocessingnya.
```bash
df.describe()
```
Disini terlihat di dalam datasetsnya terdapat 660 baris data. Cukup banyak.
### EDA
```bash
sns.heatmap(df.corr(), annot=True)
```
![download](https://github.com/dontkeep/clustering-bank-cust/assets/105641121/053f7471-5a8c-49e4-9216-30cbf9ff3ff2)<br>
Diatas merupakan heatmap untuk menunjukkan korelasi antar kolomnya. Bisa dilihat ya Total_visits_bank memiliki korelasi yang cukup tinggi dengan Total_Credit_Cards yaitu 32% dan Total_visits_online memiliki korelasi tertinggi dengan Avg_Credit_Limit. Saya tidak akan melihat pada kolom Sl_No karena itu hanyalah serial numbernya.
```bash
plt.scatter(df["Avg_Credit_Limit"], df["Total_Credit_Cards"])
plt.xlabel("Avg credit limit")
plt.ylabel("Total credit cards")
plt.show()
```
![download](https://github.com/dontkeep/clustering-bank-cust/assets/105641121/55a1488e-b187-4d2b-a586-d0a7833267c3)<br>
Diatas merupakan data yang nantinya akan kita clusterkan karena untuk membuat sebuah klasifikasi member/credit card kedua hal itulah yang paling utama. Jumlah credit card yang dimiliki dan credit limit yang dimiliki pengguna. <br>
Kita akan melihat histogram dari masing masing column yang ngga kalah penting dari 2 column diatas
```bash
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["Avg_Credit_Limit"], color="brown")
ax.set_xlabel("Avg_Credit_Limit")
plt.show()
```
![download](https://github.com/dontkeep/clustering-bank-cust/assets/105641121/b80b61f1-ec9a-471a-9b1e-f4b56205f744) <br>
Rata-rata credit limitnya berada dibawah angka 25000.
```bash
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["Total_Credit_Cards"], color="blue")
ax.set_xlabel("Total_Credit_Cards")
plt.show()
```
![download](https://github.com/dontkeep/clustering-bank-cust/assets/105641121/1f5890be-219c-468d-86c3-03d50f3c72ab)<br>
Diatas kita bisa lihat bahwa orang-orang kebanyakan memiliki 4 jumlah credit card. Diikuti dengan 6 credit card.
```bash
counts, bins, patches = ax.hist(df["Total_visits_online"], bins=10)

labels = [f"{b:.0f}-{b + (bins[1] - bins[0]):.0f}" for b in bins[:-1]]

plt.figure(figsize=(10, 10))
plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140, colors=["skyblue", "lightcoral", "gold", "limegreen", "teal"],
       wedgeprops=dict(width=0.6), radius=1.2)

plt.axis("equal")
plt.title("Distribution of Total Visits Online", fontsize=14)
plt.legend(labels, loc="center left", bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()
plt.show()
```
![download](https://github.com/dontkeep/clustering-bank-cust/assets/105641121/9c1549f3-354c-45e7-904c-4c9fd9983a2b)<br>
Bisa dilihat di atas bahwa pelanggan mayoritas hanya melakukan kunjungan secara online beberapa kali, antara 0-2 kali adalah 38.3%. <br>
Setelah itu kita akan menentukan fitur yang akan kita gunakan, yaitu Avg_Credit_Limit dan Total_Credit_Cards.
### Pre-Processing
```bash
x = df.iloc[:,[2,3]]
```
Selanjutnya kita akan mencari nilai K yang paling optimal untuk digunakan dengan cara mencari elbow point dari hasil kmeans. Berikut kodenya,
```bash
kmeans = KMeans()
inertia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(x)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, "bx-")
plt.xlabel("Number of Cluster")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimum Number of Clusters")
plt.show()
```
![download](https://github.com/dontkeep/clustering-bank-cust/assets/105641121/5cf3822a-5b17-4968-87c2-1a597a9ef703)<br>
Bisa dilihat diatas bahwa 3 merupakan nilai K nya karena perubahan dari angka 3 ke angka 4 sangat kecil sedangkan dari angka 2 ke angka 3 sangatlah significant sehingga menjadikan angka 3 ini menjadi nilai K yang paling optimal.

## Modeling
Tahap modeling ini kita gunakan KMeans dengan jumlah cluster 3 karena itu merupakan nilai yang paling optimal.
```bash
kmeans = KMeans(n_clusters=3).fit(x)
print(kmeans.cluster_centers_)
kmeans.labels_
```
Selanjutnya memasukkan hasil cluster kedalam kolom yang baru, yang bernama "cluster"
```bash
clusters_kmeans = kmeans.labels_
df["cluster"] = clusters_kmeans
df.head()
```
Dan tahap modeling pun sudah selesaii!!
### Visualisasi hasil algoritma
Membuat plot box untuk melihat Avg credit limit vs cluster dan total credit cards vs clusternya,
```bash
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))

plt.subplot(1,2,1)
sns.boxplot(data = df, x = "cluster", y = "Avg_Credit_Limit", hue = "cluster");
plt.title("Average Credit Limit vs Cluster")

plt.subplot(1,2,2)
sns.boxplot(data = df, x = "cluster", y = "Total_Credit_Cards", hue = "cluster");
plt.title("Total Credit Cards vs Cluster")

plt.show()
```
![download](https://github.com/dontkeep/clustering-bank-cust/assets/105641121/86bec85f-9586-4b3f-8a35-81fdcd4de39f)<br>
Dan langkah terakhir adalah membuat plot yang sudah meng-cluster data-data dan menunjukkan titik tengah dari masing-masing cluster yang ada.
```bash
plt.figure(figsize=(8, 6))

for label in set(clusters_kmeans):
    cluster_data = x[clusters_kmeans == label]
    plt.scatter(cluster_data['Avg_Credit_Limit'], cluster_data['Total_Credit_Cards'], label=f"Cluster {label}", alpha=0.8)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', label='Cluster Centers', marker='*')

plt.title("Cluster Plot")
plt.xlabel("Avg_Credit_Limit")
plt.ylabel("Total_Credit_Cards")

plt.legend()
plt.show()
```
![download](https://github.com/dontkeep/clustering-bank-cust/assets/105641121/8e9d5d91-33a9-4270-9b76-229275cbb411)<br>
Disini kita akan simpulkan bahwa cluster 0 bisa dibilang member Platinum, Cluster 2 member Gold, dan Cluster 1 member Silver karena merekalah yang cenderung memiliki credit limit yang rendah dan jumlah credit card yang rendah juga.
## Deployment
[Customer Segmentation Web App](https://clustering-bank-cust-mine.streamlit.app/) <br> <br>
![image](https://github.com/dontkeep/clustering-bank-cust/assets/105641121/a4e927fd-c2a0-4cca-b94a-20b5cae1a596)

