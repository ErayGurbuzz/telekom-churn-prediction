import joblib
import pandas as pd
from pathlib import Path

# --- DİZİN YAPILANDIRMASI ---
# Mevcut dosyanın konumundan (src/) bir üst klasöre (root) çıkarak models klasörüne ulaşırız.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.pkl"

# Modeli bellekte (RAM) tutmak için global bir değişken tanımlıyoruz.
# Her tahmin isteğinde diske gidip dosyayı okumak yerine, bellekteki kopyayı kullanacağız.
model_pipeline = None

def load_model():
    """
    Eğitilmiş model dosyasını (.pkl) diskten okuyup belleğe yükler.
    Singleton tasarım desenine benzer bir mantıkla çalışır: 
    Model yüklü değilse yükler, yüklüyse mevcut olanı döner.
    """
    global model_pipeline
    if MODEL_PATH.exists():
        model_pipeline = joblib.load(MODEL_PATH)
    return model_pipeline


def get_prediction(customer_dict: dict):

    """
    API'den gelen ham müşteri verisini (sözlük) alır ve tahmin üretir.
    
    Args:
        customer_dict (dict): Pydantic tarafından doğrulanmış müşteri verileri.
        
    Returns:
        dict: Tahmin sonucu, etiketi ve olasılık değeri.
    """

    global model_pipeline
    # 1. Kontrol: Model bellekte var mı? Yoksa yüklemeye çalış.
    if model_pipeline is None:
        load_model()

    # 2. Kontrol: Hala yoksa (dosya silinmiş veya bulunamamışsa) hata fırlat.
    if model_pipeline is None:
        raise ValueError("Model yüklenemedi. Önce train.py çalıştırın.")

    # Pandas DataFrame'e çeviriyoruz çünkü Scikit-learn Pipeline nesnesi bu formatı bekler.    
    input_df = pd.DataFrame([customer_dict])

    # 3. TAHMİN AŞAMASI
    # predict: Direkt olarak 0 (Kalıcı) veya 1 (Churn) sonucunu verir.
    prediction = int(model_pipeline.predict(input_df)[0])

    # predict_proba: İhtimal hesaplar. [0] Kalma ihtimali, [1] Ayrılma ihtimalidir.
    # Bizim için kritik olan [1] yani 'Ayrılma İhtimali'dir.
    probability = float(model_pipeline.predict_proba(input_df)[0][1])
    
    return {
        "prediction": prediction,
        "prediction_label": "Churn (Ayrılacak)" if prediction == 1 else "Kalıcı",
        "probability": round(probability, 4) # Daha temiz bir çıktı için 4 basamağa yuvarlıyoruz.
    }