"""Dashboard Statistik Deskriptif (versi Streamlit)
Jalankan:  streamlit run streamlit_app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(BASE, "data", "data1.csv")

st.set_page_config(page_title="Dashboard Statistik Deskriptif", page_icon="📊", layout="wide")


def f(x, d=2):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}".replace(".", ",")


# ---------------------------------------------------------------- data
def parse(raw: pd.DataFrame) -> pd.DataFrame:
    """Rapikan tabel mentah jadi angka. Menangani data1.xlsx (1 kolom berisi teks '5,5,6,...')."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all").astype(str).reset_index(drop=True)
    if raw.shape[1] == 1 and raw.iloc[:, 0].str.contains(",").all():
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
    if data.isna().all().all():
        raise ValueError("Tidak ada angka yang terbaca.")
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


@st.cache_data
def default_data() -> pd.DataFrame:
    return parse(pd.read_csv(DEFAULT, header=None, dtype=str))


# ---------------------------------------------------------------- statistik
def describe(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    vc = s.value_counts()
    mode = vc[vc == vc.max()].index.min() if (vc.max() > 1 or len(s) == 1) else np.nan
    sd = s.std(ddof=1) if len(s) > 1 else 0.0
    m = s.mean()
    return pd.Series({
        "n": len(s), "mean": m, "median": s.median(), "modus": mode, "min": s.min(), "Q1": q1, "Q3": q3,
        "max": s.max(), "range": s.max() - s.min(), "IQR": q3 - q1, "varians": sd ** 2, "std": sd,
        "CV %": sd / m * 100 if m else 0.0, "skew": s.skew() if len(s) > 2 and sd > 0 else 0.0,
    })


# ---------------------------------------------------------------- analisis otomatis
def insights(sub, cs, rs, o):
    A, D = [], []
    b = lambda t: f"**{t}**"
    cv_t = "homogen" if o["CV %"] < 20 else "cukup beragam" if o["CV %"] < 40 else "sangat beragam"
    sk = o["skew"]
    sk_t = ("hampir simetris" if abs(sk) < .5 else
            "condong ke kanan (nilai tinggi lebih jarang)" if sk > 0 else "condong ke kiri (nilai rendah lebih jarang)")
    mm = ("Mean ≈ median, jadi tidak ada pengaruh nilai ekstrem yang besar."
          if abs(o["mean"] - o["median"]) < .25 * (o["std"] or 1) else
          "Mean berbeda dari median, tanda ada pengaruh nilai ekstrem.")
    mode_t = f(o["modus"], 0) if not pd.isna(o["modus"]) else "tidak ada"
    A.append(f"{b('Gambaran umum.')} Dari {int(o['n'])} nilai: rata-rata {b(f(o['mean']))}, median {b(f(o['median'], 1))}, "
             f"modus {b(mode_t)}, SD {b(f(o['std']))}, rentang {f(o['min'], 1)}–{f(o['max'], 1)}, "
             f"Q1–Q3 = {f(o['Q1'], 1)}–{f(o['Q3'], 1)}. Distribusi {sk_t} (skewness {f(sk)}); data {cv_t} "
             f"(CV {f(o['CV %'], 1)}%). {mm}")

    cs, rs = cs.dropna(subset=["mean"]), rs.dropna(subset=["mean"])
    lo = hi = rlo = rhi = eq = None
    if len(cs) >= 2:
        bm = cs.sort_values("mean"); lo, hi = bm.index[0], bm.index[-1]
        bc = cs.sort_values("CV %")
        steps = len(cs) - 1
        d = np.sign(np.diff(cs["mean"].values))
        tr = (", dan rata-rata cenderung **naik** dari kolom ke kolom" if (d > 0).sum() / steps >= .8 else
              ", dan rata-rata cenderung **turun** dari kolom ke kolom" if (d < 0).sum() / steps >= .8 else "")
        A.append(f"{b('Antar kolom.')} Rata-rata tertinggi: {b(hi)} ({f(cs.loc[hi, 'mean'])}), terendah: {b(lo)} "
                 f"({f(cs.loc[lo, 'mean'])}){tr}. Paling stabil (CV terkecil): {b(bc.index[0])} "
                 f"({f(bc['CV %'].iloc[0], 1)}%); paling bervariasi: {b(bc.index[-1])} ({f(bc['CV %'].iloc[-1], 1)}%).")
    if len(rs) >= 2:
        bm = rs.sort_values("mean"); rlo, rhi = bm.index[0], bm.index[-1]
        t = (f"{b('Antar baris.')} Rata-rata tertinggi: {b(rhi)} ({f(rs.loc[rhi, 'mean'])}), "
             f"terendah: {b(rlo)} ({f(rs.loc[rlo, 'mean'])}).")
        groups = [g for _, g in rs.groupby(rs["mean"].round(2)) if len(g) >= 2 and g["std"].max() - g["std"].min() >= 1]
        if groups:
            eq = max(groups, key=len)
            t += (f" {len(eq)} baris ({', '.join(eq.index)}) punya rata-rata sama ({f(eq['mean'].iloc[0])}) tetapi SD "
                  f"berbeda jauh, dari {f(eq['std'].min())} sampai {f(eq['std'].max())}. "
                  f"Jadi rata-rata saja tidak cukup untuk membandingkan.")
        A.append(t)

    both = pd.concat([rs, cs])
    both = both[both["std"] > 0]
    ratio = ((both["mean"] - both["median"]).abs() / both["std"])
    gap = ratio[ratio >= .3].sort_values(ascending=False).head(5).index.tolist()
    if gap:
        A.append(f"{b('Pengaruh nilai ekstrem.')} Mean dan median cukup berjauhan di: " +
                 "; ".join(f"{n} (mean {f(both.loc[n, 'mean'])} vs median {f(both.loc[n, 'median'])})" for n in gap) +
                 ". Di sini mean tertarik nilai ekstrem.")

    ext = []
    for r in rs.index:
        if rs.loc[r, "std"] > 0:
            z = (sub.loc[r] - rs.loc[r, "mean"]) / rs.loc[r, "std"]
            ext += [(abs(v), f"{r}–{c} = {f(sub.loc[r, c], 1)}") for c, v in z.items() if pd.notna(v) and abs(v) >= 2]
    ext.sort(reverse=True)

    avg = None
    if sub.shape[1] >= 2:
        cm = sub.corr(min_periods=3)
        pairs = [(cm.iloc[i, j], cm.columns[i], cm.columns[j]) for i in range(len(cm)) for j in range(i + 1, len(cm))
                 if pd.notna(cm.iloc[i, j])]
        if pairs:
            avg = float(np.mean([p[0] for p in pairs]))
            best, worst = max(pairs), min(pairs)
            kind = "cenderung positif" if avg >= .3 else "cenderung negatif" if avg <= -.3 else "lemah"
            t = f"{b('Hubungan antar kolom.')} Rata-rata korelasi {b(f(avg))} ({kind}). Terkuat: {best[1]}–{best[2]} (r = {f(best[0])})"
            if len(pairs) > 1:
                t += f"; terlemah: {worst[1]}–{worst[2]} (r = {f(worst[0])})"
            A.append(t + ".")

    if lo:
        second = f" dan {b(cs.sort_values('mean').index[1])}" if len(cs) > 2 else ""
        D.append(f"{b('Prioritaskan perbaikan')} pada {b(lo)} (mean {f(cs.loc[lo, 'mean'])}){second}, karena nilainya paling "
                 f"rendah. Jadikan {b(hi)} (mean {f(cs.loc[hi, 'mean'])}) sebagai acuan.")
    if rlo:
        D.append(f"{b('Beri perhatian khusus')} ke {b(rlo)} (mean {f(rs.loc[rlo, 'mean'])}) sebagai yang paling lemah, dan "
                 f"jadikan {b(rhi)} (mean {f(rs.loc[rhi, 'mean'])}, SD {f(rs.loc[rhi, 'std'])}) sebagai benchmark.")
    if eq is not None:
        D.append(f"{b('Jangan menilai dari rata-rata saja.')} {', '.join(eq.index)} sama-sama bermean {f(eq['mean'].iloc[0])}, "
                 f"tapi yang SD-nya kecil bisa diperlakukan seragam, sedangkan yang SD-nya besar perlu ditinjau per kolom.")
    if gap:
        D.append(f"{b('Gunakan median, bukan mean,')} untuk {', '.join(gap)} supaya kondisi \"tipikal\" terbaca lebih adil.")
    if ext:
        D.append(f"{b('Verifikasi nilai ekstrem')} ({'; '.join(t for _, t in ext[:6])}) — salah input atau memang kejadian nyata? "
                 f"— sebelum dipakai untuk keputusan lanjutan.")
    D.append(f"{b('Tetapkan ambang berdasarkan kuartil:')} nilai di bawah Q1 ({f(o['Q1'], 1)}) ditandai perlu perbaikan, "
             f"nilai di atas Q3 ({f(o['Q3'], 1)}) ditandai unggul, dan median ({f(o['median'], 1)}) sebagai batas \"cukup\".")
    if avg is not None and abs(avg) >= .3:
        D.append(f"{b('Hubungan antar kolom')} {'positif' if avg > 0 else 'negatif'}: perubahan pada satu kolom cenderung "
                 f"diikuti kolom lain, tapi korelasi bukan sebab-akibat, jadi uji dulu sebelum mengintervensi.")
    if len(sub) < 30:
        D.append(f"{b('Hati-hati dengan sampel kecil')} ({len(sub)} baris per kolom). Kesimpulan di atas bersifat indikatif; "
                 f"tambah data sebelum keputusan besar.")
    return A, D


# ---------------------------------------------------------------- tampilan
st.title("📊 Dashboard Statistik Deskriptif")
st.caption("Versi Streamlit · edit angka di tabel, upload data sendiri, atau filter baris/kolom lewat sidebar")

if "ver" not in st.session_state:
    st.session_state.ver = 0
    st.session_state.df = default_data()
    st.session_state.fname = None

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload CSV / Excel", type=["csv", "txt", "xlsx", "xlsm"])
    if up is not None and up.file_id != st.session_state.fname:
        try:
            st.session_state.df = read_upload(up)
            st.session_state.fname = up.file_id
            st.session_state.ver += 1
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    if st.button("↺ Reset ke Data Latihan 1"):
        st.session_state.df = default_data()
        st.session_state.fname = None
        st.session_state.ver += 1
        st.rerun()

st.subheader("Data (bisa diedit)")
st.caption("Klik sel untuk mengubah angka, tambah baris lewat baris kosong di bawah tabel. Sel kosong diabaikan.")
edited = st.data_editor(st.session_state.df, num_rows="dynamic", width="stretch",
                        key=f"editor{st.session_state.ver}")
work = edited.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
work.index = [f"B{i + 1}" for i in range(len(work))]

with st.sidebar:
    st.header("Filter")
    cols = st.multiselect("Kolom", list(work.columns), default=list(work.columns))
    rows = st.multiselect("Baris", list(work.index), default=list(work.index))

sub = work.loc[rows, cols].dropna(how="all", axis=0).dropna(how="all", axis=1)
if sub.empty:
    st.warning("Pilih minimal 1 baris dan 1 kolom yang berisi angka.")
    st.stop()
st.caption(f"Dipakai: {sub.shape[0]} baris × {sub.shape[1]} kolom (dari {work.shape[0]} × {work.shape[1]})")

cs = sub.apply(describe).T
rs = sub.T.apply(describe).T
o = describe(pd.Series(sub.values.ravel()))

# KPI
k = st.columns(5)
items = [("Jumlah data", int(o["n"])), ("Mean", f(o["mean"])), ("Median", f(o["median"], 1)),
         ("Modus", f(o["modus"], 0)), ("Std. Dev", f(o["std"])), ("Varians", f(o["varians"])),
         ("Min", f(o["min"], 1)), ("Max", f(o["max"], 1)), ("IQR", f(o["IQR"], 1)), ("CV", f(o["CV %"], 1) + "%")]
for i, (label, val) in enumerate(items):
    k[i % 5].metric(label, val)

# Grafik
g1, g2 = st.columns(2)
fig = go.Figure([go.Bar(x=cs.index, y=cs["mean"], name="Mean"),
                 go.Scatter(x=cs.index, y=cs["std"], name="Std. Dev", mode="lines+markers")])
fig.update_layout(title="Rata-rata & Simpangan Baku per Kolom", yaxis_rangemode="tozero")
g1.plotly_chart(fig, width="stretch")

vals = pd.Series(sub.values.ravel()).dropna()
if (vals == vals.round()).all() and vals.max() - vals.min() <= 20:
    cnt = vals.astype(int).value_counts().reindex(range(int(vals.min()), int(vals.max()) + 1), fill_value=0)
    fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=["#ffc857" if c == cnt.max() else "#6c8cff" for c in cnt.values]))
    fig.update_xaxes(dtick=1)
else:
    fig = px.histogram(vals, nbins=10)
    fig.update_layout(showlegend=False)
fig.update_layout(title=f"Distribusi Frekuensi Seluruh Nilai (n={int(o['n'])})", xaxis_title="Nilai", yaxis_title="Frekuensi")
g2.plotly_chart(fig, width="stretch")

g3, g4 = st.columns(2)
long = sub.melt(var_name="Kolom", value_name="Nilai").dropna()
fig = px.box(long, x="Kolom", y="Nilai", category_orders={"Kolom": list(sub.columns)}, points="outliers")
fig.update_layout(title="Sebaran per Kolom (Boxplot)")
g3.plotly_chart(fig, width="stretch")

fig = go.Figure([go.Bar(x=rs.index, y=rs["mean"], name="Mean"), go.Bar(x=rs.index, y=rs["std"], name="Std. Dev")])
fig.update_layout(title="Rata-rata vs Simpangan Baku per Baris", barmode="group")
g4.plotly_chart(fig, width="stretch")

fig = px.imshow(sub, text_auto=True, aspect="auto", color_continuous_scale="Turbo")
fig.update_layout(title="Heatmap Data")
st.plotly_chart(fig, width="stretch")

# Tabel statistik
t1, t2 = st.tabs(["Statistik per Kolom", "Statistik per Baris"])
t1.dataframe(cs.round(2), width="stretch")
t2.dataframe(rs.round(2), width="stretch")

# Analisis
A, D = insights(sub, cs, rs, o)
st.header("Analisis Statistik Deskriptif")
st.caption("Dibuat otomatis dari data dan filter yang sedang aktif.")
st.markdown("\n".join(f"{i}. {t}" for i, t in enumerate(A, 1)))
st.header("Keputusan yang Bisa Diambil dari Data Ini")
st.markdown("\n".join(f"{i}. {t}" for i, t in enumerate(D, 1)))
