# 📊 Müşteri Kaybı (Churn) Analiz Raporu

---

## 1. Proje Amacı

Bu projede, telekom sektörüne ait müşteri verileri kullanılarak bir müşterinin hizmeti bırakma (churn) ihtimalini tahmin eden bir makine öğrenmesi modeli geliştirilmiştir.

Amaç:

- müşteri kaybını önceden tahmin etmek
- şirketlerin proaktif aksiyon almasını sağlamak
- modeli gerçek dünyada kullanılabilir hale getirmek (API + UI)

---

## 2. Veri Seti

Kullanılan veri seti: **Telco Customer Churn (Kaggle)**

Veri seti aşağıdaki bilgileri içerir:

- demografik bilgiler
- abonelik detayları
- hizmet kullanımı
- faturalama bilgileri
- churn (hedef değişken)

### Hedef Değişken:

- **Churn (Yes/No → 1/0)**

### Öne Çıkan Feature’lar:

- tenure
- MonthlyCharges
- TotalCharges
- Contract
- InternetService
- PaymentMethod

---

## 3. Keşifsel Veri Analizi (EDA)

### 🎯 Amaç:

Veri setindeki patternleri anlamak ve churn davranışını etkileyen faktörleri belirlemek.

---

### 📊 Temel Bulgular

#### 1. Contract Type

- **Month-to-month müşterilerde churn oranı çok yüksek**
- Uzun vadeli kontratlar churn’u ciddi şekilde azaltıyor

👉 Yorum:
Müşteri bağlılığı sözleşme süresiyle doğrudan ilişkili

---

#### 2. Tenure

- Düşük tenure → yüksek churn
- Uzun süre kalan müşteriler daha sadık

👉 Yorum:
Müşteri kaybı genellikle ilk aylarda gerçekleşiyor

---

#### 3. Monthly Charges

- Yüksek ücret → daha yüksek churn

👉 Ancak:
Tek başına fiyat belirleyici değil

---

#### 4. Payment Method

- **Electronic check kullanan müşteriler en riskli grup**

👉 Yorum:
Otomatik ödeme → daha düşük churn

---

#### 5. Internet Service

- Fiber optic kullanıcılarında churn yüksek

👉 Ek analiz:
Bu durum büyük ölçüde yüksek MonthlyCharges ile ilişkili

---

#### 6. Service Features

- OnlineSecurity ve TechSupport eksikliği churn’u artırıyor

👉 Yorum:
Hizmet kalitesi churn üzerinde kritik

---

## 4. Veri Ön İşleme

Modelleme öncesi uygulanan adımlar:

- Churn → binary (0/1)
- TotalCharges → numerik dönüşüm
- customerID → kaldırıldı
- Eksik veriler → median / most_frequent
- OneHotEncoder → kategorik değişkenler
- StandardScaler → sayısal değişkenler
- Stratified train-test split

---

## 5. Class Imbalance

Veri dağılımı:

- Churn ≈ %26

Alınan aksiyonlar:

- `class_weight="balanced"`
- `scale_pos_weight = neg/pos`

👉 Amaç:
Modelin churn sınıfını öğrenmesini sağlamak

---

## 6. Kullanılan Modeller

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

👉 Model Tournament yaklaşımı kullanıldı

---

## 7. Model Sonuçları

| Model               | Accuracy | Precision | Recall | F1         | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ---------- | ------- |
| **Random Forest**   | 0.7665   | 0.5437    | 0.7487 | **0.6299** | 0.8414  |
| XGBoost             | 0.7551   | 0.5261    | 0.7807 | 0.6286     | 0.8400  |
| CatBoost            | 0.7566   | 0.5293    | 0.7487 | 0.6202     | 0.8372  |
| LightGBM            | 0.7559   | 0.5294    | 0.7219 | 0.6109     | 0.8314  |
| Logistic Regression | 0.7381   | 0.5043    | 0.7834 | 0.6136     | 0.8413  |

---

## 8. Model Seçimi

Random Forest modeli seçildi çünkü:

- En yüksek **F1-score**
- Dengeli precision–recall
- Stabil performans

---

## 9. Model Değerlendirme

### Confusion Matrix

- churn müşterileri iyi yakalanıyor
- false positive mevcut

👉 trade-off:

- churn kaçırmamak daha önemli

---

### ROC Curve

- AUC ≈ 0.84
- model sınıfları iyi ayırıyor

---

### Precision-Recall

- recall yüksek
- precision orta

👉 model churn yakalamaya odaklı

---

## 10. Feature Importance

En önemli faktörler:

- Contract
- Tenure
- MonthlyCharges
- TotalCharges
- OnlineSecurity
- TechSupport
- PaymentMethod

👉 Model çıktıları EDA ile uyumlu

---

## 11. Business Yorumları

Bu model ile:

- churn riski yüksek müşteriler erken tespit edilir
- retention kampanyaları hedeflenebilir
- fiyatlandırma optimize edilir
- müşteri deneyimi iyileştirilebilir

---

## 12. Hyperparameter Tuning

Model performansını artırmak amacıyla RandomizedSearchCV ile hyperparameter tuning çalışmaları gerçekleştirilmiştir.

Bu süreçte:

- farklı parametre kombinasyonları test edilmiştir
- model performansı optimize edilmeye çalışılmıştır

Ancak tuning sonrası elde edilen sonuçların baseline modele kıyasla **anlamlı bir performans artışı sağlamadığı** gözlemlenmiştir.

Bu nedenle:

- model karmaşıklığını artırmamak
- daha stabil ve genellenebilir sonuçlar elde etmek

amacıyla baseline model API servisinde tercih edilmiştir.

---

## 13. Sonuç

Bu proje:

- uçtan uca ML pipeline
- güçlü model performansı
- yorumlanabilir sonuçlar
- API ile deploy edilmiş sistem

sunmaktadır.

👉 Model yalnızca tahmin yapmakla kalmaz, **iş kararlarını destekler**.
