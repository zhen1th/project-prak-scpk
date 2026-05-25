import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SPK Dosen Terbaik",
    layout="wide"
)

st.title("🎓 SPK Pemilihan Dosen Terbaik")
st.subheader("Metode AHP")

@st.cache_data
def load_data():
    df = pd.read_csv("Teaching_Quality_Evaluation_CEMP.csv")
    return df

df_raw = load_data()

df_orig = df_raw.groupby("Teacher_ID").mean(numeric_only=True).reset_index()


def ahp(matriks):
    n = matriks.shape[0]

    # ---- normalize by column ----
    col_sum = matriks.sum(axis=0)
    norm = matriks / col_sum

    # ---- priority vector ----
    w = norm.mean(axis=1)

    # ---- consistency ----
    Aw = matriks @ w
    lambda_max = np.mean(Aw / w)

    ci = (lambda_max - n) / (n - 1)

    ri_dict = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90,
        5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
        9: 1.45, 10: 1.49
    }

    ri = ri_dict.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0

    return norm, w, lambda_max, ci, cr


# Inisialisasi session state
if "criteria" not in st.session_state:
    st.session_state.criteria = []

if "teachers" not in st.session_state:
    st.session_state.teachers = []

tab0, tab1, tab2, tab3 = st.tabs([
    "📊 Dataset Mentah",
    "⚙️ Konfigurasi",
    "⚖️ Pembobotan Kriteria",
    "🏆 Ranking AHP"
])

with tab0:
    st.subheader("Dataset Mentah")
    st.dataframe(df_raw, use_container_width=True)
    st.info(f"Total baris data mentah: {len(df_raw)} baris.")

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Alternatif (Dosen)")
        all_teachers = df_orig["Teacher_ID"].tolist()
        st.session_state.teachers = st.multiselect(
            "Pilih dosen:",
            all_teachers,
            default=all_teachers[:5]
        )

    with col2:
        st.subheader("Kriteria")
        numeric_cols = df_orig.select_dtypes(include=[np.number]).columns.tolist()
        st.session_state.criteria = st.multiselect(
            "Pilih kriteria:",
            numeric_cols,
            default=numeric_cols[:5]
        )

    if st.session_state.teachers and st.session_state.criteria:
        st.write("### Data Rata-Rata Dosen Terpilih")
        st.dataframe(
            df_orig[df_orig["Teacher_ID"].isin(st.session_state.teachers)][
                ["Teacher_ID"] + st.session_state.criteria
            ],
            use_container_width=True
        )

with tab2:
    if len(st.session_state.criteria) < 2:
        st.warning("Minimal pilih 2 kriteria di tab Konfigurasi.")
    else:
        kriteria = st.session_state.criteria
        n = len(kriteria)

        st.subheader("Pairwise Kriteria")
        st.caption("Silakan tentukan tingkat kepentingan antar kriteria terlebih dahulu.")

        M = np.ones((n, n))
        scale = [
            1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2,
            1, 2, 3, 4, 5, 6, 7, 8, 9
        ]

        for i in range(n):
            for j in range(i+1, n):
                val = st.select_slider(
                    f"{kriteria[i]} vs {kriteria[j]}",
                    options=scale,
                    value=1,
                    format_func=lambda x: f"{x:.2f}"
                )
                M[i][j] = val
                M[j][i] = 1/val

        st.divider()
        
        hitung_kriteria = st.button("🔢 Hitung Bobot Kriteria", type="primary")

        if hitung_kriteria or "w_criteria" in st.session_state:
            norm, w, lam, ci, cr = ahp(M)
            
            st.session_state.w_criteria = w
            st.session_state.matriks_kriteria_terakhir = M

            st.write("### Matriks Perbandingan Kriteria")
            st.dataframe(pd.DataFrame(M, index=kriteria, columns=kriteria))

            st.write("### Hasil Bobot Prioritas Kriteria")
            st.dataframe(pd.DataFrame({
                "Kriteria": kriteria,
                "Bobot": w
            }))

            st.write("### Uji Konsistensi")
            c1, c2, c3 = st.columns(3)
            c1.metric("λ max", round(lam, 4))
            c2.metric("CI", round(ci, 4))
            c3.metric("CR", round(cr, 4))

            if cr < 0.1:
                st.success("Matriks Konsisten (CR < 0.1)")
            else:
                st.error("Matriks Tidak Konsisten (CR >= 0.1). Silakan sesuaikan kembali nilai slider Anda.")

with tab3:
    if "w_criteria" not in st.session_state:
        st.info("Silakan lakukan perhitungan di tab '⚖️ Pembobotan Kriteria' terlebih dahulu.")
    else:
        st.subheader("FULL AHP")
        st.caption("Berikan penilaian perbandingan antar dosen untuk setiap kriteria berikut.")

        teachers = st.session_state.teachers
        criteria = st.session_state.criteria

        m_alt = len(teachers)
        m_crit = len(criteria)

        W_local = np.zeros((m_crit, m_alt))
        scale = [
            1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2,
            1, 2, 3, 4, 5, 6, 7, 8, 9
        ]

        list_M_alternatif = []
        for k, crit in enumerate(criteria):
            st.write(f"#### Kriteria: **{crit}**")
            M = np.ones((m_alt, m_alt))

            for i in range(m_alt):
                for j in range(i+1, m_alt):
                    val = st.select_slider(
                        f"[{crit}] {teachers[i]} vs {teachers[j]}",
                        options=scale,
                        value=1,
                        key=f"{crit}_{i}_{j}"
                    )
                    M[i][j] = val
                    M[j][i] = 1/val
            
            list_M_alternatif.append(M)
            st.divider()

        hitung_ranking = st.button("🏆 Hitung Ranking Akhir", type="primary")

        if hitung_ranking:
            for k, crit in enumerate(criteria):
                M_curr = list_M_alternatif[k]
                _, w_alt, _, ci, cr = ahp(M_curr)
                W_local[k] = w_alt

                with st.expander(f"Lihat Detail Perhitungan Kriteria: {crit}"):
                    st.dataframe(pd.DataFrame(M_curr, index=teachers, columns=teachers))
                    st.write("Bobot Lokal Alternatif:")
                    st.dataframe(pd.DataFrame({"Teacher": teachers, "Bobot": w_alt}))
                    st.metric("CR Kriteria Ini", round(cr, 4))

            st.subheader("Hasil Akhir Perankingan")
            
            w_global = np.dot(st.session_state.w_criteria, W_local)

            result = pd.DataFrame({
                "Teacher": teachers,
                "Score": w_global
            }).sort_values("Score", ascending=False)

            result["Rank"] = range(1, len(result) + 1)
            
            st.dataframe(result.set_index("Rank"), use_container_width=True)

            st.write("### Grafik Perbandingan Skor Akhir")
            fig, ax = plt.subplots()
            ax.barh(result["Teacher"], result["Score"], color='#1f77b4')
            ax.invert_yaxis()  
            ax.set_xlabel("Total Skor Global")
            st.pyplot(fig)