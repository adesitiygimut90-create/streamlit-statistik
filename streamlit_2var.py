"""Statistik Deskriptif 2 Variabel: Korelasi, Regresi Linear & Prediksi (Streamlit)
Jalankan:  streamlit run streamlit_2var.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Statistik 2 Variabel & Prediksi", page_icon="📈", layout="wide")

# Contoh dari Excel: Pengunjung (X) vs Penjualan (Y) selama 10 hari
CONTOH = pd.DataFrame({"Pengunjung (X)": [4, 3, 5, 8, 8, 9, 7, 10, 11, 12],
                       "Penjualan (Y)": [5, 7, 7, 5, 7, 6, 6, 9, 11, 11]}, dtype=float)
CONTOH.index = range(1, 11)


def f(x, d=2):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


# ---------------------------------------------------------------- baca data
def parse(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(how="all").dropna(axis=1, how="all").astype(str).reset_index(drop=True)
    if raw.shape[1] == 1 and raw.iloc[:, 0].str.contains(",").all():  # format data1.xlsx
        raw = raw.iloc[:, 0].str.split(",", expand=True)
    num = raw.apply(lambda c: pd.to_numeric(c.str.strip().str.replace(",", ".", regex=False), errors="coerce"))
    has_header = num.iloc[0].isna().any() and (raw.iloc[0].str.strip() != "").any()
    if has_header:
        names = [("K%d" % (j + 1) if (not n.strip() or n.strip().lower().startswith("column")) else n.strip())
                 for j, n in enumerate(raw.iloc[0])]
        data = num.iloc[1:]
    else:
        names, data = ["K%d" % (j + 1) for j in range(raw.shape[1])], num
    data = data.reset_index(drop=True).astype(float)
    data.columns = names
    data = data.dropna(axis=1, how="all")
    if data.shape[1] < 2:
        raise ValueError("Butuh minimal 2 kolom angka.")
    data.index = range(1, len(data) + 1)
    return data


def read_upload(file) -> pd.DataFrame:
    if file.name.lower().endswith((".xlsx", ".xlsm")):
        raw = pd.read_excel(file, header=None)
    else:
        try:
            raw = pd.read_csv(file, header=None, sep=None, engine="python", dtype=str)
        except Exception:
            file.seek(0)
            raw = pd.read_csv(file, header=None, sep=",", dtype=str)
    return parse(raw)


# ---------------------------------------------------------------- statistik
def describe(s: pd.Series, ddof: int) -> pd.Series:
    s = s.dropna()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    vc = s.value_counts()
    mode = vc[vc == vc.max()].index.min() if vc.max() > 1 else np.nan
    sd = s.std(ddof=ddof)
    return pd.Series({"n": len(s), "mean": s.mean(), "median": s.median(), "modus": mode, "min": s.min(),
                      "Q1": q1, "Q3": q3, "max": s.max(), "range": s.max() - s.min(), "IQR": q3 - q1,
                      "varians": sd ** 2, "std": sd, "CV %": sd / s.mean() * 100 if s.mean() else np.nan,
                      "skew": s.skew()})


def strength(r):
    a = abs(r)
    return ("sangat lemah" if a < .2 else "lemah" if a < .4 else "sedang" if a < .6 else "kuat" if a < .8 else "sangat kuat")


# ---------------------------------------------------------------- sidebar & data
if "ver" not in st.session_state:
    st.session_state.ver, st.session_state.df, st.session_state.fid = 0, CONTOH.copy(), None

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload CSV / Excel", type=["csv", "txt", "xlsx", "xlsm"])
    if up is not None and up.file_id != st.session_state.fid:
        try:
            st.session_state.df, st.session_state.fid = read_upload(up), up.file_id
            st.session_state.ver += 1
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    if st.button("↺ Reset ke contoh (Pengunjung vs Penjualan)"):
        st.session_state.df, st.session_state.fid = CONTOH.copy(), None
        st.session_state.ver += 1
        st.rerun()
    st.header("Pengaturan")
    sd_type = st.radio("Simpangan baku / varians", ["Populasi (÷ n)  — sama dengan contoh Excel", "Sampel (÷ n−1)"])
    ddof = 0 if sd_type.startswith("Populasi") else 1

st.title("📈 Statistik Deskriptif 2 Variabel + Prediksi")
st.caption("Pilih variabel X (penyebab/prediktor) dan Y (yang diprediksi), lihat korelasi, persamaan regresi, lalu prediksi.")

st.subheader("1. Data (bisa diedit)")
edited = st.data_editor(st.session_state.df, num_rows="dynamic", width="stretch", key=f"ed{st.session_state.ver}")
work = edited.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
work.index = range(1, len(work) + 1)

cols = list(work.columns)
with st.sidebar:
    st.header("Variabel")
    xc = st.selectbox("Variabel X (prediktor)", cols, index=0)
    yc = st.selectbox("Variabel Y (yang diprediksi)", cols, index=1 if len(cols) > 1 else 0)

if xc == yc:
    st.warning("Pilih dua variabel yang berbeda untuk X dan Y.")
    st.stop()
d = work[[xc, yc]].dropna()
if len(d) < 3 or d[xc].std() == 0 or d[yc].std() == 0:
    st.warning("Butuh minimal 3 pasangan data, dan nilai X maupun Y tidak boleh semuanya sama.")
    st.stop()

# ---------------------------------------------------------------- hitung
x, y, n = d[xc].values, d[yc].values, len(d)
mx, my = x.mean(), y.mean()
sx, sy = x.std(ddof=ddof), y.std(ddof=ddof)
sxx, syy, sxy = ((x - mx) ** 2).sum(), ((y - my) ** 2).sum(), ((x - mx) * (y - my)).sum()
r = sxy / np.sqrt(sxx * syy)
b = sxy / sxx
a = my - b * mx
yhat = a + b * x
resid = y - yhat
se = np.sqrt((resid ** 2).sum() / (n - 2)) if n > 2 else np.nan
r2 = r ** 2
cov = sxy / (n - ddof)

st.caption(f"Dipakai {n} pasangan data. Hasil r, a, dan b sama untuk populasi maupun sampel; yang berbeda hanya SD dan varians.")

# ---------------------------------------------------------------- ringkasan
st.subheader("2. Ringkasan: Korelasi & Regresi")
k = st.columns(6)
for col, (lab, val) in zip(k, [("Korelasi (r)", f(r)), ("R² (determinasi)", f(r2)), ("Kemiringan (b)", f(b, 4)),
                               ("Intersep (a)", f(a, 4)), ("Kovarians", f(cov)), ("Galat baku (SE)", f(se))]):
    col.metric(lab, val)
st.success(f"**Persamaan regresi:  Y' = {f(a, 4)} + {f(b, 4)} X**   (≈ Y' = {f(a)} + {f(b)} X)")

st.subheader("3. Statistik Deskriptif per Variabel")
desc = pd.DataFrame({f"X: {xc}": describe(d[xc], ddof), f"Y: {yc}": describe(d[yc], ddof)}).T
st.dataframe(desc.round(2), width="stretch")

# ---------------------------------------------------------------- prediksi
st.subheader("4. Prediksi")
c1, c2 = st.columns([1, 2])
with c1:
    xn = st.number_input(f"Nilai {xc} yang ingin diprediksi", value=float(np.ceil(x.max() + 1)), step=1.0)
    multi = st.text_input("Atau beberapa nilai X (pisahkan koma)", placeholder="contoh: 5, 10, 15")
    yn = a + b * xn
    st.metric(f"Prediksi {yc}", f(yn))
    st.caption(f"Perkiraan kisaran (±1 galat baku): {f(yn - se)} sampai {f(yn + se)}")
    if xn < x.min() or xn > x.max():
        st.warning(f"X = {f(xn, 1)} di luar rentang data ({f(x.min(), 1)}–{f(x.max(), 1)}). Ini ekstrapolasi, hasilnya kurang bisa diandalkan.")
    extra = []
    if multi.strip():
        try:
            extra = [float(v.strip().replace(",", ".")) for v in multi.replace(";", ",").split(",") if v.strip()]
        except ValueError:
            st.error("Isi daftar X dengan angka, dipisah koma.")
    if extra:
        st.dataframe(pd.DataFrame({xc: extra, f"Prediksi {yc}": [a + b * v for v in extra]}).round(2), width="stretch", hide_index=True)

with c2:
    xs = np.array([min(x.min(), xn, *extra), max(x.max(), xn, *extra)])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data", marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name=f"Y' = {f(a)} + {f(b)}X"))
    px_, py_ = [xn] + extra, [yn] + [a + b * v for v in extra]
    fig.add_trace(go.Scatter(x=px_, y=py_, mode="markers", name="Prediksi",
                             marker=dict(size=14, symbol="diamond", color="#ffc857", line=dict(width=1, color="white"))))
    fig.update_layout(title=f"{yc} vs {xc}", xaxis_title=xc, yaxis_title=yc)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- tabel hitung
with st.expander("5. Tabel perhitungan (seperti di Excel)"):
    zx, zy = (x - mx) / sx, (y - my) / sy
    tab = pd.DataFrame({"X": x, "Y": y, "X − x̄": x - mx, "(X − x̄)²": (x - mx) ** 2, "Y − ȳ": y - my,
                        "(Y − ȳ)²": (y - my) ** 2, "Zx": zx, "Zy": zy, "Zx·Zy": zx * zy, "Y' (prediksi)": yhat,
                        "Residual (Y − Y')": resid}, index=d.index.astype(str))
    tot = pd.DataFrame({"(X − x̄)²": [sxx], "(Y − ȳ)²": [syy], "Zx·Zy": [(zx * zy).sum()]}, index=["Σ"])
    st.dataframe(pd.concat([tab, tot]).round(2), width="stretch")
    st.markdown(f"""
- Rata-rata: x̄ = **{f(mx)}**, ȳ = **{f(my)}** · SD: sX = **{f(sx)}**, sY = **{f(sy)}** ({'populasi' if ddof == 0 else 'sampel'})
- Korelasi: r = Σ(Zx·Zy) / {'n' if ddof == 0 else '(n−1)'} = {f((zx * zy).sum())} / {n if ddof == 0 else n - 1} = **{f(r, 4)}**
- Kemiringan: b = r · (sY / sX) = {f(r, 4)} × ({f(sy)} / {f(sx)}) = **{f(b, 4)}**
- Intersep: a = ȳ − b · x̄ = {f(my)} − {f(b, 4)} × {f(mx)} = **{f(a, 4)}**
- Persamaan: **Y' = a + bX = {f(a, 4)} + {f(b, 4)}X**
""")

# ---------------------------------------------------------------- interpretasi
st.subheader("6. Interpretasi & Keputusan")
arah = "positif (X naik → Y cenderung naik)" if r > 0 else "negatif (X naik → Y cenderung turun)"
pts = [
    f"**Hubungan {strength(r)} dan {arah}**, r = {f(r)}.",
    f"**Setiap kenaikan 1 unit {xc}**, {yc} rata-rata {'naik' if b > 0 else 'turun'} sekitar **{f(abs(b))}** unit.",
    f"**R² = {f(r2)}**, artinya sekitar {f(r2 * 100, 1)}% variasi {yc} bisa dijelaskan oleh {xc}; sisanya {f((1 - r2) * 100, 1)}% dipengaruhi faktor lain.",
    f"**Ketepatan prediksi:** rata-rata selisih prediksi dengan data asli sekitar {f(se)} unit (galat baku).",
    f"**Intersep {f(a)}** adalah prediksi {yc} saat {xc} = 0; hanya bermakna jika X = 0 masuk akal dan ada dalam rentang data.",
]
if r2 >= .5:
    pts.append(f"**Keputusan:** {xc} cukup layak dipakai untuk memperkirakan {yc}; upaya menaikkan {xc} kemungkinan ikut mendorong {yc}.")
else:
    pts.append(f"**Keputusan:** {xc} saja belum cukup kuat untuk memprediksi {yc}. Pertimbangkan variabel tambahan sebelum dipakai untuk keputusan penting.")
if n < 30:
    pts.append(f"**Hati-hati:** data hanya {n} pasangan, jadi persamaan ini indikatif. Korelasi juga bukan bukti sebab-akibat, dan jangan memprediksi di luar rentang X ({f(x.min(), 1)}–{f(x.max(), 1)}).")
st.markdown("\n".join(f"{i}. {t}" for i, t in enumerate(pts, 1)))
