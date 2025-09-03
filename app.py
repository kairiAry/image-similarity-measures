import streamlit as st
import cv2
import numpy as np
import os


def binerisasi_otsu(image):
    """Mengubah citra menjadi biner menggunakan metode Otsu."""
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, biner_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return biner_image

def extract_feature(biner_image):
    """Mengekstrak fitur sederhana (jumlah piksel putih) dari citra biner."""
    if biner_image is None:
        return 0
    return np.sum(biner_image == 255)

def euclidean_distance(feature_a, feature_b):
    """Menghitung jarak Euclidean antara dua fitur (satu dimensi)."""
    return np.sqrt((feature_a - feature_b)**2)

def load_database_from_folder(folder_path):
    """Membaca semua gambar dari folder yang ditentukan."""
    image_database = {}
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return image_database
        
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                image_database[filename] = img
    return image_database

st.title("Aplikasi Klasifikasi Gambar dengan Metode Otsu")
st.write("Program ini mengklasifikasikan gambar berdasarkan jumlah piksel putih (luas objek) setelah binerisasi menggunakan metode Otsu.")

database_folder = 'database_images'
image_database = load_database_from_folder(database_folder)

if not image_database:
    st.warning(f"Tidak ada gambar ditemukan di folder '{database_folder}'. Silakan tambahkan gambar ke folder tersebut.")
else:
    uploaded_file = st.file_uploader("Pilih sebuah gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_compare = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gambar Asli")
            st.image(image_to_compare, channels="BGR", caption="Gambar yang Diunggah")
        
        biner_img_to_compare = binerisasi_otsu(image_to_compare)
        with col2:
            st.subheader("Hasil Binerisasi")
            st.image(biner_img_to_compare, channels="GRAY", caption="Citra Biner")

        feature_to_compare = extract_feature(biner_img_to_compare)
        st.write(f"**Fitur (jumlah piksel putih):** {feature_to_compare}")

        st.write("---")
        st.subheader("Hasil Perbandingan")
        
        best_match = {"filename": None, "distance": float('inf')}
        
        for filename, db_image in image_database.items():
            biner_db_image = binerisasi_otsu(db_image)
            feature_db = extract_feature(biner_db_image)
            
            distance = euclidean_distance(feature_to_compare, feature_db)
            
            st.write(f"Jarak Euclidean dengan **{filename}**: `{distance:.2f}`")
            
            if distance < best_match["distance"]:
                best_match["distance"] = distance
                best_match["filename"] = filename
                best_match["image"] = db_image

        st.write("---")
        
        if best_match["filename"] is not None:
            st.success(f"Gambar yang paling mirip adalah **{best_match['filename']}** dengan Jarak: `{best_match['distance']:.2f}`")
            st.subheader("Gambar Paling Mirip")
            st.image(best_match["image"], channels="BGR", caption=best_match["filename"])
            if best_match["filename"] == "indomi.jpg":
                st.subheader("Gambar diatas adalah Indomie Goreng seleraku😋😋")
            if best_match["filename"] == "deodorant.jpg":
                st.subheader("Gambar diatas adalah Deodorant")
            if best_match["filename"] == "mouse.jpg":
                st.subheader("Gambar diatas adalah Mouse")
            if best_match["filename"] == "pulpen.jpg":
                st.subheader("Gambar diatas adalah Pulpen")
            if best_match["filename"] == "tipex.jpg":
                st.subheader("Gambar diatas adalah Tipe-x")
            if best_match["filename"] == "wancuh.jpg":
                st.subheader("Gambar diatas adalah Sendok nasi")
        else:
            st.error("Tidak dapat menemukan kecocokan yang valid.")