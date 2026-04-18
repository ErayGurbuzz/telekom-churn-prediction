import streamlit as st
import requests

# --- SAYFA YAPILANDIRMASI ---
# Tarayıcı sekmesinde görünecek başlık ve sayfa düzeni (centered: ortalanmış)
st.set_page_config(
    page_title="Müşteri Kaybı Analiz Paneli", 
    page_icon="📊", 
    layout="centered"
)

# Ana Başlık ve Açıklama
st.title("📊 Telekom Müşteri Kaybı (Churn) Tahmin Aracı")
st.markdown("""
Bu arayüz, arka planda çalışan **FastAPI** servisimize bağlanarak makine öğrenmesi destekli tahminler üretir. 
Müşteri bilgilerini girerek ayrılma riskini anlık olarak hesaplayabilirsiniz.
""")

# --- VERİ GİRİŞ FORMU ---
# Form yapısı, her veri değişiminde sayfanın en baştan yüklenmesini engelleyerek daha akıcı bir deneyim sunar.
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demografik Bilgiler")
        gender = st.selectbox("Cinsiyet", ["Male", "Female"])
        # Kullanıcı dostu seçim (Evet/Hayır), API'ye 1/0 olarak gönderilir.
        senior_secim = st.selectbox("Yaşlı Müşteri mi?", ["Hayır", "Evet"])
        senior = 1 if senior_secim == "Evet" else 0
        tenure = st.number_input("Şirkette Geçirdiği Ay (Tenure)", min_value=0, value=12)
        contract = st.selectbox("Sözleşme Tipi", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.number_input("Aylık Ücret ($)", min_value=0.0, value=70.5)
        
    with col2:
        st.subheader("Hizmet Detayları")
        internet_service = st.selectbox("İnternet Servisi", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Ödeme Yöntemi", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        total_charges = st.number_input("Toplam Ödenen Ücret ($)", min_value=0.0, value=850.0)
        
        # Kolaylık olsun diye diğer tüm özellikleri varsayılan olarak ayarlıyoruz (İstersen ekleyebilirsin)
        st.info("💡 Diğer özellikler (Güvenlik, Yedekleme vb.) analiz için varsayılan değerlerde işleme alınacaktır.")
    # Hesaplama Butonu
    submit = st.form_submit_button("Ayrılma İhtimalini Hesapla")

# --- API İLETİŞİMİ VE SONUÇ EKRANI ---
if submit:
    # FastAPI'ye gönderilecek JSON formatındaki veri
    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    # Arka plandaki API'mize istek atıyoruz
    try:
        # Backend API'ye POST isteği gönderiyoruz
        with st.spinner('Yapay zeka modeli analiz ediyor...'):
            response = requests.post("http://127.0.0.1:8000/predict", json=data)

        if response.status_code == 200:
            result = response.json()
            st.markdown("---")

            # Risk durumuna göre görsel geri bildirim (Kırmızı/Yeşil)
            prob_percent = result['probability'] * 100

            if result['prediction'] == 1:
                st.error(f"⚠️ **RİSKLİ MÜŞTERİ:** Bu müşterinin ayrılma (Churn) ihtimali **%{prob_percent:.2f}**")
                st.progress(result['probability']) # Görsel bar ekleme
            else:
                st.success(f"✅ **GÜVENLİ MÜŞTERİ:** Bu müşterinin ayrılma ihtimali sadece **%{prob_percent:.2f}**")
                st.balloons() # Başarı kutlaması efekti
        else:
            st.warning("⚠️ API yanıt verdi ancak tahmin hatası oluştu. Lütfen veri girişlerini kontrol edin.")
    except requests.exceptions.ConnectionError:
        st.error("❌ API Bağlantı Hatası: Lütfen arka planda FastAPI (Uvicorn) servisinin çalıştığından emin olun.")
    except Exception as e:
        st.error(f"❌ Beklenmedik bir hata oluştu: {e}")