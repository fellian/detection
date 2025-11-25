import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(
    page_title="Edge Detection & Contour Highlighting", layout="wide", page_icon="✨"
)

# Custom CSS
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf9 100%);
        font-family: 'Inter', sans-serif;
        color: #2d3748;
        padding-top: 0;
    }

    /* Header */
    .main-header {
        background: linear-gradient(120deg, #3a7bd5, #00d2ff);
        color: white;
        text-align: center;
        padding: 2.2rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 15s infinite linear;
    }
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        max-width: 700px;
        margin: 0 auto;
        font-weight: 400;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
        padding: 1.5rem;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #3a7bd5 !important;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }
    .stSidebar .stSelectbox, .stSidebar .stSlider {
        margin-bottom: 1.2rem;
    }

    /* Step Cards */
    .step-card {
        background: white;
        border-radius: 14px;
        padding: 1.1rem 0.8rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.35s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .step-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    .step-card i {
        font-size: 1.8rem;
        margin-bottom: 0.6rem;
        color: #3a7bd5;
    }
    .step-card b {
        font-size: 0.95rem;
        color: #2d3748;
        font-weight: 600;
        margin-top: 0.3rem;
        display: block;
    }

    /* Images */
    .stImage > img {
        border-radius: 12px;
        border: 1px solid #edf2f7;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    /* Buttons */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(120deg, #3a7bd5, #00d2ff);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem 1.4rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(58, 123, 213, 0.3);
        width: auto;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(58, 123, 213, 0.45);
    }

    /* Horizontal Rule */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #cbd5e0, transparent);
        margin: 1.8rem 0;
    }

    /* Histograms */
    .stPlotlyChart, .stPyplot {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    }
    .matplotlib-figure {
        background: transparent !important;
    }
    .matplotlib-text {
        color: #2d3748 !important;
    }

    /* Upload prompt */
    .stAlert {
        background-color: #ebf4ff;
        border: 1px dashed #a0c4ff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: #3a7bd5;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
    <h1><i class="fas fa-border-style"></i> Edge Detection & Contour Highlighting</h1>
    <p>Visualisasi interaktif Gaussian Blur, Sobel/Canny Edge Detection, Dilation, dan Overlay Kontur</p>
</div>
""",
    unsafe_allow_html=True,
)

# Upload banyak gambar
uploaded_files = st.file_uploader(
    "📂 Unggah satu atau beberapa gambar (jpg/png/jpeg)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info(
        "⬆️ Silakan unggah satu atau lebih gambar untuk memulai analisis deteksi tepi."
    )
else:
    # Sidebar tetap berlaku global untuk semua gambar
    st.sidebar.header("⚙️ Pengaturan Parameter")
    blur_kernel = st.sidebar.slider("Ukuran Gaussian Blur", 3, 15, 5, step=2)
    method = st.sidebar.selectbox("Metode Deteksi Tepi", ["Canny", "Sobel"])
    dilate_iter = st.sidebar.slider("Dilation Iterations", 1, 5, 1)

    if method == "Canny":
        low_th = st.sidebar.slider("Canny Threshold Bawah", 0, 255, 100)
        high_th = st.sidebar.slider("Canny Threshold Atas", 0, 255, 200)

    st.sidebar.subheader("🎨 Warna Overlay")
    color_hex = st.sidebar.color_picker("Pilih Warna Overlay", "#00FF00")
    overlay_color = tuple(int(color_hex[i : i + 2], 16) for i in (1, 3, 5))

    # Loop tiap file
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### 🖼️ Gambar {idx + 1}: `{uploaded_file.name}`")
        with st.expander(
            f"Lihat proses deteksi tepi untuk `{uploaded_file.name}`", expanded=True
        ):
            try:
                image = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(image)

                # Proses
                blur = cv2.GaussianBlur(img_array, (blur_kernel, blur_kernel), 0)

                if method == "Canny":
                    edges = cv2.Canny(blur, low_th, high_th)
                else:
                    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    edges = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))

                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(edges, kernel, iterations=dilate_iter)
                overlay = img_array.copy()
                overlay[dilated != 0] = overlay_color

                # Visualisasi Tahapan
                st.subheader("📊 Tahapan Proses Deteksi Tepi")
                cols = st.columns(5)
                step_icons = [
                    "fa-image",
                    "fa-circle-notch",
                    "fa-wave-square",
                    "fa-expand",
                    "fa-layer-group",
                ]
                step_labels = [
                    "Original",
                    "Gaussian Blur",
                    f"Edge ({method})",
                    "Dilation",
                    f"Overlay ({color_hex})",
                ]

                for i, (col, icon, label, img, is_gray) in enumerate(
                    zip(
                        cols,
                        step_icons,
                        step_labels,
                        [image, blur, edges, dilated, overlay],
                        [False, False, True, True, False],
                    )
                ):
                    with col:
                        st.markdown(
                            f"""
                        <div class="step-card">
                            <i class="fas {icon}"></i>
                            <b>{label}</b>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                        if is_gray:
                            st.image(img, use_container_width=True, channels="GRAY")
                        else:
                            st.image(img, use_container_width=True)

                # Histogram
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader("📈 Histogram Sebelum & Sesudah")

                gray_original = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY)

                col_hist1, col_hist2 = st.columns(2)

                with col_hist1:
                    st.markdown("**Histogram Sebelum (Original)**")
                    fig1, ax1 = plt.subplots(facecolor="none")
                    ax1.hist(
                        gray_original.ravel(),
                        bins=256,
                        range=[0, 256],
                        color="#3a7bd5",
                        alpha=0.8,
                    )
                    ax1.set_title("Original", fontsize=12, color="#2d3748")
                    ax1.set_xlabel("Intensitas Piksel")
                    ax1.set_ylabel("Frekuensi")
                    ax1.spines["top"].set_visible(False)
                    ax1.spines["right"].set_visible(False)
                    st.pyplot(fig1, transparent=True)

                with col_hist2:
                    st.markdown("**Histogram Sesudah (Overlay)**")
                    fig2, ax2 = plt.subplots(facecolor="none")
                    ax2.hist(
                        gray_overlay.ravel(),
                        bins=256,
                        range=[0, 256],
                        color="#00d2ff",
                        alpha=0.8,
                    )
                    ax2.set_title("Overlay", fontsize=12, color="#2d3748")
                    ax2.set_xlabel("Intensitas Piksel")
                    ax2.set_ylabel("Frekuensi")
                    ax2.spines["top"].set_visible(False)
                    ax2.spines["right"].set_visible(False)
                    st.pyplot(fig2, transparent=True)

                # Perbandingan Before–After
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader(f"🔀 Perbandingan Before–After – Gambar {idx + 1}")
                image_comparison(
                    img1=np.array(image),
                    img2=overlay,
                    label1="Original",
                    label2="Overlay Result",
                    width=min(900, st.session_state.get("max_width", 900)),
                    starting_position=50,
                )

                # Unduh hasil
                st.markdown("<hr>", unsafe_allow_html=True)
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode(".png", overlay_bgr)
                st.download_button(
                    label=f"💾 Unduh Hasil Overlay – {uploaded_file.name}",
                    data=buffer.tobytes(),
                    file_name=f"edge_overlay_{idx+1}_{color_hex.replace('#','')}.png",
                    mime="image/png",
                    key=f"download_{idx}",
                )

            except Exception as e:
                st.error(f"❌ Gagal memproses `{uploaded_file.name}`: {str(e)}")

        st.markdown("<hr style='margin: 2.5rem 0;'>", unsafe_allow_html=True)
