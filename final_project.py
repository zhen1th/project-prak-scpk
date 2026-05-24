import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SPK Dosen Terbaik - AHP Saaty",
    layout="wide"
)

st.title("🎓 SPK Pemilihan Dosen Terbaik")
st.subheader("Metode AHP (Saaty Scale - Akademik)")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Teaching_Quality_Evaluation_CEMP.csv")

    df = df.groupby("Teacher_ID").mean(numeric_only=True).reset_index()
    return df

df_orig = load_data()

# =========================================================
# AHP FUNCTION (CORE)
# =========================================================
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


# =========================================================
# SESSION STATE
# =========================================================
if "criteria" not in st.session_state:
    st.session_state.criteria = []

if "teachers" not in st.session_state:
    st.session_state.teachers = []

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "⚙️ Konfigurasi",
    "⚖️ Pembobotan Kriteria",
    "🏆 Ranking AHP"
])

# =========================================================
# TAB 1 - CONFIG
# =========================================================
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
        st.dataframe(
            df_orig[df_orig["Teacher_ID"].isin(st.session_state.teachers)][
                ["Teacher_ID"] + st.session_state.criteria
            ],
            use_container_width=True
        )

# =========================================================
# TAB 2 - KRITERIA AHP
# =========================================================
with tab2:

    if len(st.session_state.criteria) < 2:
        st.warning("Minimal 2 kriteria")
    else:

        kriteria = st.session_state.criteria
        n = len(kriteria)

        st.subheader("Pairwise Kriteria (Saaty Scale)")

        M = np.ones((n, n))

        scale = [
            1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,
            1,2,3,4,5,6,7,8,9
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

        norm, w, lam, ci, cr = ahp(M)

        st.session_state.w_criteria = w

        st.dataframe(pd.DataFrame(M, index=kriteria, columns=kriteria))

        st.write("### Bobot Kriteria")
        st.dataframe(pd.DataFrame({
            "Kriteria": kriteria,
            "Bobot": w
        }))

        c1, c2, c3 = st.columns(3)
        c1.metric("λ max", round(lam,4))
        c2.metric("CI", round(ci,4))
        c3.metric("CR", round(cr,4))

        if cr < 0.1:
            st.success("Konsisten")
        else:
            st.error("Tidak konsisten")

# =========================================================
# TAB 3 - FULL AHP
# =========================================================
with tab3:

    if "w_criteria" not in st.session_state:
        st.info("Isi kriteria dulu")
    else:

        st.subheader("FULL AHP (Saaty Alternatif)")

        teachers = st.session_state.teachers
        criteria = st.session_state.criteria

        df = df_orig[df_orig["Teacher_ID"].isin(teachers)].copy()

        m_alt = len(teachers)
        m_crit = len(criteria)

        W_local = np.zeros((m_crit, m_alt))

        scale = [
            1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,
            1,2,3,4,5,6,7,8,9
        ]

        for k, crit in enumerate(criteria):

            st.divider()
            st.subheader(f"Kriteria: {crit}")

            M = np.ones((m_alt, m_alt))

            for i in range(m_alt):
                for j in range(i+1, m_alt):

                    val = st.select_slider(
                        f"{teachers[i]} vs {teachers[j]}",
                        options=scale,
                        value=1,
                        key=f"{crit}_{i}_{j}"
                    )

                    M[i][j] = val
                    M[j][i] = 1/val

            _, w_alt, _, ci, cr = ahp(M)

            W_local[k] = w_alt

            st.dataframe(pd.DataFrame(M, index=teachers, columns=teachers))

            st.write("Bobot Alternatif")
            st.dataframe(pd.DataFrame({
                "Teacher": teachers,
                "Bobot": w_alt
            }))

            st.metric("CR", round(cr,4))

        # ==========================
        # GLOBAL SCORE
        # ==========================
        st.divider()
        st.subheader("Ranking Akhir")

        w_global = np.dot(st.session_state.w_criteria, W_local)

        result = pd.DataFrame({
            "Teacher": teachers,
            "Score": w_global
        }).sort_values("Score", ascending=False)

        result["Rank"] = range(1, len(result)+1)

        st.dataframe(result)

        fig, ax = plt.subplots()
        ax.barh(result["Teacher"], result["Score"])
        ax.invert_yaxis()
        ax.set_xlabel("Score")
        st.pyplot(fig)