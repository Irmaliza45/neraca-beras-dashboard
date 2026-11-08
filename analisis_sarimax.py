"""
Analisis SARIMAX neraca beras Kabupaten Bojonegoro (metode Box-Jenkins).

Modul ini tidak bergantung pada Streamlit, sehingga bisa dipakai oleh app.py
maupun dijalankan sendiri:  python analisis_sarimax.py

Model yang dipakai
------------------
1. Model utama  : SARIMAX(1,0,0)(0,1,0)12 untuk ln(produksi beras)
                  dengan variabel eksogen X = ln(luas panen).
                    ln P_t = beta * ln L_t + u_t
                    (1 - phi1 B)(1 - B^12) u_t = e_t
2. Model untuk X: SARIMA([2],0,0)(0,1,0)12 untuk ln(luas panen), dipakai
                  untuk meramal luas panen tahun berikutnya (ramalan ex ante).
                    (1 - phi2 B^2)(1 - B^12) ln L_t = e_t
3. Pembanding   : SARIMA(2,0,0)(1,1,0)12 untuk ln(produksi) tanpa X.
4. Konsumsi     : ARIMA(0,1,0) dengan drift.

Orde model tidak ditulis mati: fungsi `kandidat()` menghitung ulang semua
kandidat, lalu `pilih_terbaik()` memilih yang AIC-nya terkecil di antara
kandidat yang seluruh parameternya signifikan dan residualnya lolos Ljung-Box.
"""

import hashlib
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

warnings.filterwarnings("ignore")

S = 12          # panjang satu siklus musim (data bulanan)
ALPHA = 0.05    # taraf signifikansi

# Cache hasil seleksi model. Mengestimasi lebih dari 100 kandidat SARIMAX butuh waktu lama di server,
# jadi hasilnya disimpan di file ini. Cache otomatis dihitung ulang bila data atau aturan kandidat berubah.
CACHE = Path(__file__).parent / "hasil" / "cache_seleksi_model.json"
VERSI_KANDIDAT = "p,q in 0..2; P,Q in 0..1; d=0; D=1; s=12; tambahan luas [2]; v1"


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def siapkan_seri(df):
    """Mengubah tabel bulanan menjadi deret waktu dengan indeks tanggal."""
    df = df.sort_values(["Tahun", "Bulan_No"]).reset_index(drop=True)
    idx = pd.to_datetime(dict(year=df["Tahun"], month=df["Bulan_No"], day=1))
    idx = pd.DatetimeIndex(idx, freq="MS")
    return {
        "idx": idx,
        "P": pd.Series(df["Produksi_Beras_Ton"].astype(float).values, idx, name="Produksi"),
        "L": pd.Series(df["Luas_Panen_Ha"].astype(float).values, idx, name="Luas"),
        "K": pd.Series(df["Konsumsi_Beras_Ton"].astype(float).values, idx, name="Konsumsi"),
    }


# ---------------------------------------------------------------------
# Tahap 1: identifikasi
# ---------------------------------------------------------------------
def uji_stasioner(x):
    """ADF (H0: tidak stasioner) dan KPSS (H0: stasioner)."""
    x = pd.Series(x).dropna().values
    adf = adfuller(x, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = kpss(x, regression="c", nlags="auto")
    return {"n": len(x), "adf_stat": adf[0], "adf_p": adf[1], "kpss_stat": kp[0], "kpss_p": kp[1],
            "stasioner": (adf[1] < ALPHA) and (kp[1] > ALPHA)}


def beda_musiman(x, s=S):
    """(1 - B^s) X_t = X_t - X_{t-s}"""
    return (x - x.shift(s)).dropna()


def korelogram(x, nlags):
    x = pd.Series(x).dropna().values
    nlags = min(nlags, len(x) - 1)
    nlags_p = min(nlags, len(x) // 2 - 1)
    return {"acf": acf(x, nlags=nlags), "pacf": pacf(x, nlags=nlags_p, method="ywm"),
            "batas": 1.96 / np.sqrt(len(x)), "n": len(x)}


# ---------------------------------------------------------------------
# Tahap 2: estimasi kandidat dan seleksi model
# ---------------------------------------------------------------------
def fit(y, order, sorder, exog=None):
    return SARIMAX(y, exog=exog, order=order, seasonal_order=tuple(sorder) + (S,)).fit(disp=False)


def jumlah_arma(order, sorder):
    hitung = lambda v: len(v) if isinstance(v, (list, tuple)) else int(v)
    return hitung(order[0]) + hitung(order[2]) + hitung(sorder[0]) + hitung(sorder[2])


def label_orde(order, sorder):
    teks = lambda v: f"[{','.join(map(str, v))}]" if isinstance(v, (list, tuple)) else str(v)
    return f"({teks(order[0])},{order[1]},{teks(order[2])})({sorder[0]},{sorder[1]},{sorder[2]})₁₂"


def kandidat(y, exog=None, D=1, tambahan=()):
    """Mengestimasi semua kombinasi p,q in {0,1,2}, P,Q in {0,1} (d=0, D tetap)."""
    daftar = [((p, 0, q), (P, D, Q)) for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1])]
    daftar += list(tambahan)
    baris = []
    for order, sorder in daftar:
        try:
            m = fit(y, order, sorder, exog)
        except Exception:
            continue
        r = m.resid.iloc[S * D:]
        k = jumlah_arma(order, sorder)
        lb = acorr_ljungbox(r, lags=[6], model_df=min(k, 5), return_df=True)["lb_pvalue"].iloc[0]
        pv = m.pvalues.drop("sigma2", errors="ignore")
        baris.append({
            "Model": label_orde(order, sorder), "order": order, "sorder": sorder,
            "Jumlah_parameter": len(m.params), "AIC": m.aic, "BIC": m.bic,
            "LjungBox_p": lb, "Semua_signifikan": bool((pv < ALPHA).all()) if len(pv) else True,
            "p_maks": float(pv.max()) if len(pv) else 0.0,
            "Konvergen": bool(m.mle_retvals.get("converged", True)),
        })
    return pd.DataFrame(baris).sort_values("AIC").reset_index(drop=True)


def pilih_terbaik(tabel):
    """AIC terkecil di antara kandidat yang memenuhi syarat Box-Jenkins."""
    layak = tabel[tabel["Semua_signifikan"] & (tabel["LjungBox_p"] > ALPHA) & tabel["Konvergen"]
                  & (tabel["Jumlah_parameter"] > 1)]
    return (layak if len(layak) else tabel).iloc[0]


# ---------------------------------------------------------------------
# Tahap 3: diagnosis residual
# ---------------------------------------------------------------------
def diagnosis(model, order, sorder, buang=S):
    """Uji residual sesuai materi: ACF residual, Ljung-Box, normalitas, ACF residual kuadrat (efek ARCH).
    Residual awal sebanyak d + D*s dibuang karena belum bermakna setelah diferensiasi."""
    r = model.resid.iloc[buang:]
    k = jumlah_arma(order, sorder)
    lb = acorr_ljungbox(r, lags=[6, 12], model_df=min(k, 5), return_df=True)
    sw = stats.shapiro(r)
    jb = stats.jarque_bera(r)
    arch = het_arch(r, nlags=4)
    uji = pd.DataFrame([
        ["Non-autokorelasi", "Ljung-Box lag 6", lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0]],
        ["Non-autokorelasi", "Ljung-Box lag 12", lb["lb_stat"].iloc[1], lb["lb_pvalue"].iloc[1]],
        ["Normalitas", "Shapiro-Wilk", sw.statistic, sw.pvalue],
        ["Normalitas", "Jarque-Bera", jb.statistic, jb.pvalue],
        ["Tidak ada efek ARCH", "ARCH-LM (lag 4)", arch[0], arch[1]],
    ], columns=["Asumsi", "Uji", "Statistik", "p_value"])
    uji["Terpenuhi"] = uji["p_value"] > ALPHA
    return {"resid": r, "uji": uji, "acf": acf(r, nlags=12), "acf2": acf(r ** 2, nlags=12),
            "batas": 1.96 / np.sqrt(len(r))}


def tabel_koefisien(model, nama_exog="ln(Luas panen)"):
    ci = model.conf_int()
    nama = {"ar.L1": "φ₁ (AR lag 1)", "ar.L2": "φ₂ (AR lag 2)", "ma.L1": "θ₁ (MA lag 1)", "ma.L2": "θ₂ (MA lag 2)",
            "ar.S.L12": "Φ₁ (AR musiman lag 12)", "ma.S.L12": "Θ₁ (MA musiman lag 12)", "sigma2": "σ² (varians inovasi)"}
    return pd.DataFrame({
        "Parameter": [nama.get(i, f"β ({nama_exog})" if i not in nama else i) for i in model.params.index],
        "Kode": model.params.index,
        "Koefisien": model.params.values, "Std_error": model.bse.values,
        "z": model.tvalues.values, "p_value": model.pvalues.values,
        "Bawah_95": ci.iloc[:, 0].values, "Atas_95": ci.iloc[:, 1].values,
    })


# ---------------------------------------------------------------------
# Konsumsi: ARIMA(0,1,0) dengan drift
# ---------------------------------------------------------------------
def model_konsumsi(K):
    dK = K.diff().dropna()
    m = SARIMAX(dK, order=(0, 0, 0), trend="c").fit(disp=False)
    return m, float(m.params["intercept"])


def ramal_konsumsi(K, h, drift):
    return K.iloc[-1] + drift * np.arange(1, h + 1)


# ---------------------------------------------------------------------
# Peramalan ex ante
# ---------------------------------------------------------------------
def ramal_ex_ante(m_prod, m_luas, h=12, geser_luas=0.0):
    """Ramalan produksi tanpa mengetahui luas panen masa depan.
    1) luas panen diramal dengan SARIMA, 2) hasilnya dimasukkan sebagai X ke SARIMAX.
    Varians ramalan = varians SARIMAX (bersyarat X) + beta^2 x varians ramalan ln(luas panen).
    geser_luas: penyesuaian skenario terhadap ramalan luas panen, dalam persen."""
    fl = m_luas.get_forecast(h)
    ln_luas = fl.predicted_mean.values + np.log1p(geser_luas / 100)
    fp = m_prod.get_forecast(h, exog=ln_luas.reshape(-1, 1))
    beta = float(m_prod.params.iloc[0])
    var = fp.var_pred_mean.values + beta ** 2 * fl.var_pred_mean.values
    return pd.DataFrame({
        "mu": fp.predicted_mean.values, "sd": np.sqrt(var),
        "ln_luas": ln_luas, "sd_luas": np.sqrt(fl.var_pred_mean.values),
    }, index=fl.predicted_mean.index)


def evaluasi_holdout(seri, order_p, sorder_p, order_l, sorder_l, order_u, sorder_u, n_uji=12):
    """Latih pada data selain 12 bulan terakhir, lalu ramal 12 bulan terakhir."""
    lnP, lnL, P = np.log(seri["P"]), np.log(seri["L"]), seri["P"]
    latih, uji = slice(0, len(P) - n_uji), slice(len(P) - n_uji, len(P))
    aktual = P.iloc[uji]
    hasil = {}
    hasil["Naif musiman (sama dengan tahun lalu)"] = P.iloc[len(P) - 2 * n_uji:len(P) - n_uji].values
    mu = fit(lnP.iloc[latih], order_u, sorder_u)
    hasil[f"SARIMA tanpa X {label_orde(order_u, sorder_u)}"] = np.exp(mu.get_forecast(n_uji).predicted_mean.values)
    mp = fit(lnP.iloc[latih], order_p, sorder_p, exog=lnL.iloc[latih])
    ml = fit(lnL.iloc[latih], order_l, sorder_l)
    lnL_hat = ml.get_forecast(n_uji).predicted_mean.values
    hasil["SARIMAX ex ante (luas panen diramal)"] = np.exp(mp.get_forecast(n_uji, exog=lnL_hat.reshape(-1, 1)).predicted_mean.values)
    hasil["SARIMAX ex post (luas panen aktual)"] = np.exp(mp.get_forecast(n_uji, exog=lnL.iloc[uji].values.reshape(-1, 1)).predicted_mean.values)
    baris = []
    for nama, pred in hasil.items():
        e = aktual.values - pred
        baris.append({"Metode": nama, "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
                      "MAPE": np.mean(np.abs(e) / aktual.values) * 100})
    return pd.DataFrame(baris), pd.DataFrame(hasil, index=aktual.index).assign(Aktual=aktual.values)


def contoh_hitung(m_prod, m_luas, seri):
    """Substitusi manual ramalan satu bulan ke depan (seperti contoh pada artikel SARIMA)."""
    lnP, lnL = np.log(seri["P"]), np.log(seri["L"])
    n = len(lnP)
    beta = float(m_prod.params.iloc[0])
    phi1 = float(m_prod.params.get("ar.L1", 0.0))
    phi2 = float(m_luas.params.get("ar.L2", 0.0))
    phi1_luas = float(m_luas.params.get("ar.L1", 0.0))
    W = lnL - lnL.shift(S)                         # (1 - B^12) ln L
    W_hat = phi1_luas * W.iloc[n - 1] + phi2 * W.iloc[n - 2]
    lnL_hat = lnL.iloc[n - S] + W_hat
    u = lnP - beta * lnL                           # galat regresi
    Z = u - u.shift(S)                             # (1 - B^12) u
    Z_hat = phi1 * Z.iloc[n - 1]
    u_hat = u.iloc[n - S] + Z_hat
    lnP_hat = beta * lnL_hat + u_hat
    return {
        "beta": beta, "phi1": phi1, "phi2": phi2,
        "lnL_t12": lnL.iloc[n - S], "W_t1": W.iloc[n - 2], "W_t": W.iloc[n - 1], "W_hat": W_hat,
        "lnL_hat": lnL_hat, "L_hat": np.exp(lnL_hat),
        "u_t12": u.iloc[n - S], "Z_t": Z.iloc[n - 1], "Z_hat": Z_hat, "u_hat": u_hat,
        "lnP_hat": lnP_hat, "P_hat": np.exp(lnP_hat),
        "L_t12": seri["L"].iloc[n - S], "P_t12": seri["P"].iloc[n - S],
        "lab_t12": seri["idx"][n - S], "lab_t": seri["idx"][n - 1], "lab_t1": seri["idx"][n - 2],
    }


# ---------------------------------------------------------------------
# Cache seleksi model
# ---------------------------------------------------------------------
def _sidik_data(seri):
    """Sidik jari data: berubah bila satu angka pun di data berubah."""
    isi = np.concatenate([seri["P"].values, seri["L"].values, seri["K"].values]).astype(float).tobytes()
    tanggal = ",".join(seri["idx"].strftime("%Y-%m")).encode()
    return hashlib.md5(VERSI_KANDIDAT.encode() + tanggal + isi).hexdigest()


def _ke_json(tabel):
    return json.loads(tabel.to_json(orient="records"))


def _dari_json(baris):
    t = pd.DataFrame(baris)
    t["order"] = t["order"].apply(tuple)
    t["sorder"] = t["sorder"].apply(tuple)
    return t


def muat_atau_hitung_kandidat(seri, lnP, lnL, paksa=False):
    sidik = _sidik_data(seri)
    if not paksa:
        try:
            simpan = json.loads(CACHE.read_text(encoding="utf-8"))
            if simpan.get("sidik") == sidik:
                return (_dari_json(simpan["kand_prod"]), _dari_json(simpan["kand_luas"]),
                        _dari_json(simpan["kand_uni"]))
        except Exception:
            pass
    kand_prod = kandidat(lnP, exog=lnL)
    kand_luas = kandidat(lnL, tambahan=[(([2], 0, 0), (0, 1, 0)), (([2], 0, 0), (1, 1, 0)), (([2], 0, 0), (0, 1, 1))])
    kand_uni = kandidat(lnP)
    try:
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps({"sidik": sidik, "versi": VERSI_KANDIDAT, "kand_prod": _ke_json(kand_prod),
                                     "kand_luas": _ke_json(kand_luas), "kand_uni": _ke_json(kand_uni)}, indent=1),
                         encoding="utf-8")
    except Exception:
        pass
    return kand_prod, kand_luas, kand_uni


# ---------------------------------------------------------------------
# Jalankan semua
# ---------------------------------------------------------------------
def jalankan(df):
    seri = siapkan_seri(df)
    lnP, lnL = np.log(seri["P"]), np.log(seri["L"])

    identifikasi = {
        "ln_P": uji_stasioner(lnP), "ln_P_D12": uji_stasioner(beda_musiman(lnP)),
        "ln_L": uji_stasioner(lnL), "ln_L_D12": uji_stasioner(beda_musiman(lnL)),
        "kor_ln_P": korelogram(lnP, 18), "kor_ln_P_D12": korelogram(beda_musiman(lnP), 12),
        "kor_ln_L_D12": korelogram(beda_musiman(lnL), 12),
    }

    kand_prod, kand_luas, kand_uni = muat_atau_hitung_kandidat(seri, lnP, lnL)
    best_p, best_l, best_u = pilih_terbaik(kand_prod), pilih_terbaik(kand_luas), pilih_terbaik(kand_uni)

    m_prod = fit(lnP, best_p["order"], best_p["sorder"], exog=lnL)
    m_luas = fit(lnL, best_l["order"], best_l["sorder"])
    m_uni = fit(lnP, best_u["order"], best_u["sorder"])
    m_kons, drift = model_konsumsi(seri["K"])

    evaluasi, ramalan_uji = evaluasi_holdout(seri, best_p["order"], best_p["sorder"], best_l["order"],
                                             best_l["sorder"], best_u["order"], best_u["sorder"])
    return {
        "seri": seri, "identifikasi": identifikasi,
        "kand_prod": kand_prod, "kand_luas": kand_luas, "kand_uni": kand_uni,
        "best_prod": best_p, "best_luas": best_l, "best_uni": best_u,
        "m_prod": m_prod, "m_luas": m_luas, "m_uni": m_uni, "m_kons": m_kons, "drift": drift,
        "diag_prod": diagnosis(m_prod, best_p["order"], best_p["sorder"]),
        "diag_luas": diagnosis(m_luas, best_l["order"], best_l["sorder"]),
        "diag_uni": diagnosis(m_uni, best_u["order"], best_u["sorder"]),
        "koef_prod": tabel_koefisien(m_prod), "koef_luas": tabel_koefisien(m_luas), "koef_uni": tabel_koefisien(m_uni),
        "evaluasi": evaluasi, "ramalan_uji": ramalan_uji,
        "contoh": contoh_hitung(m_prod, m_luas, seri),
    }


if __name__ == "__main__":
    from pathlib import Path
    data = pd.read_csv(Path(__file__).parent / "data" / "data_neraca_beras_bersih.csv")
    H = jalankan(data)
    print("Model produksi :", H["best_prod"]["Model"], "AIC", round(H["best_prod"]["AIC"], 2))
    print("Model luas     :", H["best_luas"]["Model"], "AIC", round(H["best_luas"]["AIC"], 2))
    print("Pembanding     :", H["best_uni"]["Model"], "AIC", round(H["best_uni"]["AIC"], 2))
    print(H["koef_prod"].round(4).to_string(index=False))
    print(H["diag_prod"]["uji"].round(4).to_string(index=False))
    print(H["evaluasi"].round(1).to_string(index=False))
    R = ramal_ex_ante(H["m_prod"], H["m_luas"])
    print("Total ramalan produksi:", round(np.exp(R["mu"]).sum()))
    c = H["contoh"]
    print("Contoh hitung bulan pertama:", round(c["P_hat"]), "vs statsmodels", round(np.exp(R["mu"].iloc[0])))
