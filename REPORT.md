# VERİ BİLİMİ CHALLENGE RAPORU

## PROJE ADI

Müşteri Kaybı Tahmini ve Basit API Servisi

## 1. Proje Amacı

Bu proje kapsamında Telco Customer Churn veri seti kullanılarak bir müşterinin hizmeti bırakıp bırakmayacağını tahmin ederek, şirketin proaktif önlemler almasını sağlayacak karar destek sistemi oluşturmayı amaçlar.bir makine öğrenmesi modeli geliştirilmiştir. Çalışmanın amacı yalnızca model eğitmek değil, aynı zamanda bu modeli HTTP üzerinden erişilebilir bir API servisi haline getirerek uçtan uca çalışan bir sistem oluşturmaktır.

## 2. Veri Seti Hakkında

Projede Kaggle üzerinde yer alan Telco Customer Churn veri seti kullanılmıştır. Veri seti, telekom sektöründeki müşterilere ait demografik bilgiler, abonelik detayları, faturalama bilgileri ve müşterinin hizmeti bırakıp bırakmadığını gösteren hedef değişkeni içermektedir.

Hedef değişken:

- `Churn` → Müşterinin hizmeti bırakıp bırakmadığını gösterir.

Öne çıkan değişkenler:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `Contract`
- `InternetService`
- `PaymentMethod`

## 3. Keşifsel Veri Analizi

Veri seti incelendiğinde hem sayısal hem de kategorik değişkenler içerdiği görülmüştür. Analiz sürecinde aşağıdaki başlıklara odaklanılmıştır:

- Hedef değişken dağılımı
- Kategorik değişkenlerin churn ile ilişkisi
- Sayısal değişkenlerin dağılımı
- Eksik veya hatalı veri kontrolü

## 4. Veri Ön İşleme

Modelleme öncesinde verinin kalite ve tutarlılığını artırmak için aşağıdaki adımlar uygulanmıştır:

- **Hedef Değişken Dönüşümü:** `Churn` değişkeni `Yes/No` formatından makine öğrenmesi modellerinin anlayabileceği `1/0` (binary) formatına çevrilmiştir.
- **Tip Dönüşümü:** `TotalCharges` sütunundaki numerik olmayan hatalı karakterler temizlenmiş ve sütun veri tipi problemli olduğu görüldüğü için numerik veri tipine dönüştürülmüştür.
- **Boyut Azaltma:** `customerID` sütunu tahminleme için istatistiksel bir anlam taşımadığı için veri setinden çıkarılmıştır.
- **Eksik Veri Yönetimi:** Veri setindeki eksik değerler (NaN), veri dağılımını bozmamak adına `median` ve `most_frequent` stratejileri ile doldurulmuştur.
- **Kategorik Kodlama:** Kategorik değişkenler, algoritmaların işleyebilmesi için `OneHotEncoder` yöntemiyle sayısal vektörlere dönüştürülmüştür.
- **Ölçeklendirme:** Sayısal değişkenlerin birbirine üstünlük sağlamaması için `StandardScaler` (Z-score normalization) uygulanmıştır.
- **Veri Bölme:** Veri eğitim ve test olarak ayrılırken, hedef değişkenin (Churn) sınıflar arası dağılımını her iki sette de korumak amacıyla `stratify` parametresi kullanılmıştır.

## 5. Sınıf Dengesizliği (Class Imbalance) Çözümü

Veri setinde churn eden müşterilerin oranı düşüktür (%26). Modelin bu azınlık grubu yakalayabilmesi (Recall başarısı) için:

- Geleneksel modellerde `class_weight="balanced"` parametresi kullanılmıştır.
- Boosting modellerinde dinamik `scale_pos_weight` (Negatif Sınıf / Pozitif Sınıf oranı) hesaplanarak modellere enjekte edilmiştir.

## 6. Denenen Modeller

Projede **"Model Tournament" (Model Yarıştırma)** stratejisi uygulanmıştır. Tek bir model yerine 5 farklı algoritma (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) aynı veri seti üzerinde test edilmiş ve en yüksek **F1-Score**'u veren model prodüksiyon için seçilmiştir.

Bu çalışma kapsamında beş farklı model denenmiştir:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost

## 7. Model Değerlendirme

Modeller aşağıdaki metriklerle değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Bu problemde yalnızca accuracy değerine bakmak yeterli değildir.

> `metrics.json` dosyasındaki gerçek sonuçlar

| Model             | Accuracy   | Precision | Recall     | F1-Score   | ROC-AUC    |
| ----------------- | ---------- | --------- | ---------- | ---------- | ---------- |
| **Random Forest** | **0.7665** | 0.5437    | 0.7487     | **0.6299** | **0.8414** |
| XGBoost           | 0.7551     | 0.5261    | 0.7807     | 0.6286     | 0.8400     |
| Logistic Reg.     | 0.7381     | 0.5043    | **0.7834** | 0.6136     | 0.8413     |

## 8. Seçilen Model ve Gerekçesi

Karşılaştırma sonucunda **Random Forest** modeli, hem doğruluk hem de ayrılanları yakalama (Recall) dengesinde en yüksek F1-Score'u (0.6299) vererek şampiyon seçilmiştir.

## 9. API Servisi

Eğitilen model, FastAPI kullanılarak servis haline getirilmiştir. API içerisinde en az bir adet `/predict` endpoint’i bulunmaktadır.

### Endpoint

- `POST /predict`

Ek olarak sistem durumu kontrolü için `/health` endpoint’i de eklenmiştir.

## 10. Mimari Tasarım

- **FastAPI:** Model, tip güvenliği sağlayan (Pydantic) ve yüksek performanslı bir API arkasında servis edilmektedir.
- **Streamlit:** Karar vericiler için anlık görsel geri bildirim sunan, risk durumuna göre renkli uyarılar (Success/Error) veren bir arayüz geliştirilmiştir.
- **Persistence:** Tüm preprocessing ve model adımları tek bir `joblib` pipeline dosyasında birleştirilerek taşınabilirlik sağlanmıştır.

## 11. Arayüzsüz Örnek Kullanım

Örnek istek:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1050.2
}
```

Örnek çıktı:

```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "probability": 0.8123
}
```

## 12. Sonuç

Bu proje kapsamında müşteri kaybını tahmin etmeye yönelik bir makine öğrenmesi çözümü geliştirilmiş ve bu çözüm API servisi haline getirilmiştir. Çalışma, veri temizleme, ön işleme, modelleme, değerlendirme ve servisleştirme adımlarını kapsayan uçtan uca bir yaklaşım sunmaktadır.

Geliştirilen sistem, bir müşterinin ayrılma riskini sadece tahmin etmekle kalmaz, aynı zamanda bu riskin olasılığını (probability) yüzde bazında sunarak işletmeye önceliklendirme yapma imkanı tanır.
