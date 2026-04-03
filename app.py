# =============================================================================
# HOTEL GEO-ANALYTICS SUITE — v2.0
# Streamlit App — Phase 1 (données fictives + upload CSV réel)
# =============================================================================
# Déploiement : streamlit run app_v2.py
# Cloud       : push sur GitHub → Streamlit Community Cloud
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import HeatMap, MiniMap
from streamlit_folium import st_folium
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
from libpysal.weights import DistanceBand
from esda.moran import Moran, Moran_Local
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler
import io
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde, linregress
from libpysal.weights import DistanceBand
from esda.moran import Moran, Moran_Local
from sklearn.preprocessing import StandardScaler
import warnings

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Geo-Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1a1a2e; }
[data-testid="stSidebar"] * { color: #e8e8e0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #c2c0b6 !important; }
[data-testid="stSidebar"] hr { border-color: #333 !important; }
.metric-card {
    background: white;
    border: 0.5px solid #e8e6e0;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 4px;
}
.insight-box {
    background: #EAF3DE;
    border-left: 3px solid #3B6D11;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 14px;
}
.alert-box {
    background: #FCEBEB;
    border-left: 3px solid #A32D2D;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 14px;
}
.info-box {
    background: #E6F1FB;
    border-left: 3px solid #185FA5;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
DEC_COLORS = {
    "HOLD ★": "#2E8B57", "CAPEX": "#4682B4",
    "REPOSITIONNER": "#FF8C00", "CÉDER": "#DC143C", "SURVEILLER": "#808080",
}
LISA_COLORS = {
    "HH": "#E84242", "LL": "#4472C4",
    "HL": "#FF8C00", "LH": "#9370DB", "NS": "#CCCCCC",
}
LISA_LABELS = {
    "HH": "Hot spot — fort ADR groupé",
    "LL": "Cold spot — faible ADR groupé",
    "HL": "Opportunité (fort dans zone faible)",
    "LH": "Risque (faible dans zone forte)",
    "NS": "Non-significatif",
}

REQUIRED_COLS = ["lat", "lon", "adr", "stars", "rooms"]

# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_demo_data():
    np.random.seed(42)
    zones = {
        "Centre Affaires":     dict(lat=48.860, lon=2.350, slat=0.010, slon=0.013, adr=195, n=80),
        "Quartier Historique": dict(lat=48.854, lon=2.341, slat=0.008, slon=0.010, adr=175, n=60),
        "District Loisirs":    dict(lat=48.841, lon=2.368, slat=0.012, slon=0.015, adr=155, n=50),
        "Banlieue Nord":       dict(lat=48.878, lon=2.346, slat=0.015, slon=0.018, adr=108, n=40),
        "Zone Émergente":      dict(lat=48.836, lon=2.372, slat=0.013, slon=0.016, adr=122, n=35),
        "Périphérie":          dict(lat=48.832, lon=2.338, slat=0.018, slon=0.020, adr=78,  n=35),
    }
    records = []
    for zone, cfg in zones.items():
        for _ in range(cfg["n"]):
            lat   = np.random.normal(cfg["lat"], cfg["slat"])
            lon   = np.random.normal(cfg["lon"], cfg["slon"])
            stars = np.random.choice([2,3,4,5], p=[0.10,0.35,0.38,0.17])
            rooms = int(np.clip(np.random.lognormal(4.4, 0.5), 20, 450))
            adr   = max(50, cfg["adr"] + (stars-3)*32 + np.random.normal(0,20))
            occ   = np.clip(np.random.normal(0.73,0.09), 0.45, 0.96)
            records.append(dict(lat=round(lat,6), lon=round(lon,6),
                                stars=stars, rooms=rooms,
                                adr=round(adr,0), occ=round(occ,3),
                                revpar=round(adr*occ,0), zone=zone))
    return pd.DataFrame(records)

@st.cache_data
def load_demo_portfolio():
    rows = [
        ("Grand Central",   48.860,2.348,5,180,310,0.77,"HOLD ★",   45),
        ("Le Quartier",     48.855,2.341,4,120,210,0.77,"REPOSITIONNER",28),
        ("Station Nord",    48.872,2.355,3, 95,130,0.70,"SURVEILLER",12),
        ("Business Park A", 48.863,2.368,4,200,175,0.80,"HOLD ★",   35),
        ("Loft & Suites",   48.848,2.360,4, 85,195,0.80,"HOLD ★",   18),
        ("Le Belvédère",    48.850,2.335,5,140,280,0.80,"HOLD ★",   52),
        ("Arena Hotel",     48.840,2.365,3, 75,115,0.70,"CÉDER",     8),
        ("Terminus Hôtel",  48.866,2.345,2, 60, 80,0.65,"CAPEX",     5),
        ("Les Jardins",     48.843,2.352,3,110,140,0.70,"CAPEX",    15),
        ("Sky Lounge",      48.858,2.377,4,160,225,0.81,"REPOSITIONNER",32),
        ("Portail Sud",     48.834,2.342,2, 50, 70,0.66,"CÉDER",     4),
        ("Académie Hôtel",  48.861,2.356,3, 90,125,0.70,"CAPEX",    11),
    ]
    df = pd.DataFrame(rows, columns=["nom","lat","lon","stars","rooms","adr","occ","decision","valeur"])
    df["revpar"] = (df["adr"] * df["occ"]).round(0)
    return df

def validate_upload(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes : {missing}. Requises : {REQUIRED_COLS}")
        return None
    df = df.dropna(subset=REQUIRED_COLS)
    if "occ" not in df.columns:
        df["occ"] = 0.72
    if "revpar" not in df.columns:
        df["revpar"] = (df["adr"] * df["occ"]).round(0)
    if "zone" not in df.columns:
        df["zone"] = "Non défini"
    return df

@st.cache_data
def compute_lisa(df_hash, adr_values, lats, lons):
    coords = list(zip(lons, lats))
    try:
        w = DistanceBand(coords, threshold=0.025, binary=False)
        w.transform = "r"
        ml = Moran_Local(adr_values, w, permutations=499)
        sig = ml.p_sim < 0.05
        q   = ml.q
        return np.where(~sig,"NS",np.where(q==1,"HH",np.where(q==3,"LL",np.where(q==4,"HL","LH"))))
    except Exception:
        return np.full(len(adr_values), "NS")

@st.cache_data
def compute_gwr(df_hash, y_vals, X_vals, lons, lats):
    coords = np.array(list(zip(lons, lats)))
    y      = np.array(y_vals).reshape(-1,1)
    X_sc   = StandardScaler().fit_transform(np.array(X_vals))
    X_c    = np.hstack([np.ones((len(y),1)), X_sc])
    try:
        bw  = Sel_BW(coords, y, X_c, kernel="bisquare").search()
        res = GWR(coords, y, X_c, bw=bw, kernel="bisquare").fit()
        return bw, res.params, res.localR2, res.predy.flatten(), res.resid_response.flatten()
    except Exception as e:
        st.error(f"Erreur GWR : {e}")
        return None, None, None, None, None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## 🏨 Hotel Geo-Analytics")
        st.markdown("*Real Estate Intelligence Suite*")
        st.divider()

        # Source données
        st.markdown("### Données")
        data_mode = st.radio("Source", ["Mode démo", "Upload CSV"], label_visibility="collapsed")

        df_market    = load_demo_data()
        df_portfolio = load_demo_portfolio()

        if data_mode == "Upload CSV":
            st.markdown("**Marché (hôtels)**")
            f_market = st.file_uploader("CSV marché", type="csv", key="mkt",
                                         help="Colonnes requises : lat, lon, adr, stars, rooms")
            if f_market:
                try:
                    up = pd.read_csv(f_market)
                    validated = validate_upload(up)
                    if validated is not None:
                        df_market = validated
                        st.success(f"✓ {len(df_market)} hôtels chargés")
                except Exception as e:
                    st.error(f"Erreur lecture : {e}")

            st.markdown("**Portefeuille**")
            f_port = st.file_uploader("CSV portefeuille", type="csv", key="port",
                                       help="Colonnes : nom, lat, lon, adr, stars, rooms, decision")
            if f_port:
                try:
                    up = pd.read_csv(f_port)
                    if "nom" in up.columns:
                        df_portfolio = up
                        st.success(f"✓ {len(df_portfolio)} actifs chargés")
                except Exception as e:
                    st.error(f"Erreur : {e}")

        st.divider()

        # Filtres globaux
        st.markdown("### Filtres marché")
        stars_filter = st.multiselect("Étoiles", [2,3,4,5], default=[2,3,4,5])
        adr_range    = st.slider("ADR (€)", int(df_market["adr"].min()),
                                 int(df_market["adr"].max()),
                                 (int(df_market["adr"].min()), int(df_market["adr"].max())))
        if "zone" in df_market.columns and df_market["zone"].nunique() > 1:
            zones_all    = df_market["zone"].unique().tolist()
            zones_filter = st.multiselect("Zones", zones_all, default=zones_all)
        else:
            zones_filter = df_market["zone"].unique().tolist() if "zone" in df_market.columns else []

        # Appliquer filtres
        mask = (df_market["stars"].isin(stars_filter) &
                df_market["adr"].between(adr_range[0], adr_range[1]))
        if zones_filter and "zone" in df_market.columns:
            mask &= df_market["zone"].isin(zones_filter)
        df_market_filtered = df_market[mask].copy()

        st.caption(f"{len(df_market_filtered)} / {len(df_market)} hôtels affichés")
        st.divider()

        # Navigation
        page = st.radio("Navigation", [
            "📊 Dashboard",
            "📍 GWR + Scoring ADR manquant",
            "🔗 LISA Clustering",
            "🗺️ Carte interactive",
            "💼 Portefeuille",
            "🎯 Screening acquisition",
            "📥 Export insights",
        ], label_visibility="collapsed")

        st.divider()
        st.caption("v2.0 — Phase 1 MVP")
        st.caption("Données fictives 'Metropolis'")

    return page, df_market_filtered, df_portfolio


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 : DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard(df, portfolio):
    st.title("Dashboard — Vue marché")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Hôtels analysés",  len(df))
    c2.metric("ADR moyen",        f"€{df['adr'].mean():.0f}")
    c3.metric("RevPAR moyen",     f"€{df['revpar'].mean():.0f}")
    c4.metric("Occupation moy.",  f"{df['occ'].mean()*100:.0f}%")
    c5.metric("Actifs portefeuille", len(portfolio))

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="adr", nbins=30, color="stars",
                           title="Distribution ADR par classement",
                           labels={"adr":"ADR (€)","count":"Hôtels"},
                           color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, x="stars", y="revpar", color="stars",
                     title="RevPAR par classement étoiles",
                     labels={"revpar":"RevPAR (€)","stars":"Étoiles"})
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if "zone" in df.columns:
            zone_adr = df.groupby("zone")["adr"].mean().sort_values(ascending=True)
            fig = px.bar(zone_adr, orientation="h",
                         title="ADR moyen par zone",
                         labels={"value":"ADR moyen (€)","zone":"Zone"},
                         color=zone_adr.values,
                         color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        dec_counts = portfolio["decision"].value_counts()
        fig = px.pie(values=dec_counts.values, names=dec_counts.index,
                     color=dec_counts.index, color_discrete_map=DEC_COLORS,
                     title="Décisions portefeuille")
        st.plotly_chart(fig, use_container_width=True)

    # Tableau résumé
    st.subheader("Résumé marché par zone")
    if "zone" in df.columns:
        summary = df.groupby("zone").agg(
            n=("adr","count"), adr_moy=("adr","mean"),
            revpar_moy=("revpar","mean"), occ_moy=("occ","mean")
        ).round(1)
        summary.columns = ["N hôtels","ADR moy (€)","RevPAR moy (€)","Occ. moy"]
        st.dataframe(summary.sort_values("ADR moy (€)", ascending=False),
                     use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 : GWR + SCORING ADR MANQUANT
# ─────────────────────────────────────────────────────────────────────────────
def page_gwr(df):
    st.title("GWR — Régression spatialement pondérée + Scoring ADR manquant")

    st.markdown("""
    <div class="info-box">
    <b>Module A1 :</b> Compare l'ADR réel de chaque hôtel à l'ADR prédit par sa localisation (GWR).
    Un écart négatif = sous-performance inexpliquée → levier de management ou opportunité d'acquisition décotée.
    </div>
    """, unsafe_allow_html=True)

    if len(df) < 30:
        st.warning("Minimum 30 hôtels requis pour le GWR. Élargissez vos filtres.")
        return

    with st.spinner("Calcul GWR en cours... (15–30s selon le dataset)"):
        bw, params, local_r2, predy, resid = compute_gwr(
            hash(tuple(df["adr"].values[:10])),
            df["adr"].values,
            df[["stars","rooms","occ"]].values,
            df["lon"].values, df["lat"].values
        )

    if bw is None:
        return

    df = df.copy()
    df["beta_stars"] = params[:,1]
    df["beta_rooms"]  = params[:,2]
    df["local_r2"]   = local_r2
    df["adr_predit"]  = predy
    df["adr_gap"]     = df["adr"] - df["adr_predit"]
    df["adr_gap_pct"] = (df["adr_gap"] / df["adr_predit"] * 100).round(1)

    p20 = df["adr_gap_pct"].quantile(0.20)
    p80 = df["adr_gap_pct"].quantile(0.80)
    df["profil"] = df["adr_gap_pct"].apply(
        lambda v: "Sur-performer (+)" if v >= p80
        else ("Sous-performer (−)" if v <= p20 else "Dans la norme")
    )

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Bande passante GWR", f"{bw:.0f} voisins")
    c2.metric("R² moyen", f"{local_r2.mean():.3f}")
    c3.metric("Sous-performers", (df["profil"]=="Sous-performer (−)").sum(),
              delta=f"gap moy {df[df['profil']=='Sous-performer (−)']['adr_gap_pct'].mean():.1f}%")
    c4.metric("Sur-performers",  (df["profil"]=="Sur-performer (+)").sum(),
              delta=f"gap moy +{df[df['profil']=='Sur-performer (+)']['adr_gap_pct'].mean():.1f}%")

    st.divider()

    PROF_COLORS = {"Sur-performer (+)":"#2E8B57","Dans la norme":"#AAAAAA","Sous-performer (−)":"#DC143C"}

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x="lon", y="lat", color="profil",
                         color_discrete_map=PROF_COLORS,
                         size="rooms", size_max=18,
                         hover_data={"adr":True,"adr_predit":":.0f","adr_gap_pct":":.1f"},
                         title="Carte des profils de performance relative")
        fig.update_layout(xaxis_title="Longitude", yaxis_title="Latitude")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df, x="adr_predit", y="adr",
                         color="profil", color_discrete_map=PROF_COLORS,
                         title="ADR réel vs. prédit GWR",
                         labels={"adr_predit":"ADR prédit (€)","adr":"ADR réel (€)"})
        max_val = max(df["adr"].max(), df["adr_predit"].max()) + 20
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(dash="dash", color="black", width=1))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="adr_gap_pct", nbins=30, color="profil",
                           color_discrete_map=PROF_COLORS,
                           title="Distribution des gaps ADR (%)")
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df, x="lon", y="lat", color="local_r2",
                         color_continuous_scale="RdYlGn",
                         title="Qualité locale du modèle (R²)",
                         labels={"local_r2":"R²"})
        st.plotly_chart(fig, use_container_width=True)

    # Insight
    sous = df[df["profil"]=="Sous-performer (−)"]
    if len(sous) > 0:
        st.markdown(f"""
        <div class="alert-box">
        <b>Sous-performers détectés ({len(sous)} hôtels)</b><br>
        Gap moyen : <b>{sous['adr_gap_pct'].mean():.1f}%</b> sous leur potentiel de localisation.<br>
        Ces actifs sous-performent par rapport à ce que leur adresse devrait permettre.
        → Cause possible : management, positionnement produit, rénovation nécessaire.
        </div>
        """, unsafe_allow_html=True)

    # Table top 10 sous-performers
    st.subheader("Top 10 sous-performers (à investiguer)")
    cols_show = ["zone","stars","rooms","adr","adr_predit","adr_gap_pct","profil"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df.nsmallest(10,"adr_gap_pct")[cols_show].round(1),
                 use_container_width=True)

    st.session_state["gwr_df"] = df


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 : LISA
# ─────────────────────────────────────────────────────────────────────────────
def page_lisa(df):
    st.title("LISA — Clusters spatiaux d'ADR")

    st.markdown("""
    <div class="info-box">
    <b>Local Moran's I :</b> Classifie chaque hôtel selon son ADR et celui de ses voisins.
    Les outliers HL (fort dans zone faible) sont les opportunités d'acquisition les plus précieuses.
    </div>
    """, unsafe_allow_html=True)

    if len(df) < 20:
        st.warning("Minimum 20 hôtels requis. Élargissez les filtres.")
        return

    lisa = compute_lisa(
        hash(tuple(df["adr"].values[:10])),
        df["adr"].values, df["lat"].values, df["lon"].values
    )
    df = df.copy()
    df["lisa"] = lisa

    counts = df["lisa"].value_counts()
    cols = st.columns(5)
    for i, lt in enumerate(["HH","LL","HL","LH","NS"]):
        n = counts.get(lt, 0)
        cols[i].metric(lt, n, delta=f"{n/len(df)*100:.0f}%",
                       delta_color="off" if lt=="NS" else "normal")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for lt in ["HH","LL","HL","LH","NS"]:
            sub = df[df["lisa"]==lt]
            if len(sub)==0: continue
            fig.add_trace(go.Scatter(
                x=sub["lon"], y=sub["lat"], mode="markers",
                name=f"{lt} — {LISA_LABELS[lt][:25]}",
                marker=dict(size=8 if lt!="NS" else 5,
                            color=LISA_COLORS[lt],
                            opacity=0.85 if lt!="NS" else 0.3,
                            line=dict(width=0.5, color="white")),
                text=sub["adr"].astype(str)+"€",
                hovertemplate="<b>%{text}</b><br>"+LISA_LABELS.get(lt,"")+"<extra></extra>"
            ))
        fig.update_layout(title="Carte LISA", xaxis_title="Longitude",
                          yaxis_title="Latitude", legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df[df["lisa"]!="NS"], x="lisa", y="adr",
                     color="lisa", color_discrete_map=LISA_COLORS,
                     title="Distribution ADR par cluster LISA",
                     labels={"lisa":"Type LISA","adr":"ADR (€)"})
        st.plotly_chart(fig, use_container_width=True)

    # Insights stratégiques
    hl = df[df["lisa"]=="HL"]
    hh = df[df["lisa"]=="HH"]
    ll = df[df["lisa"]=="LL"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="insight-box">
        <b>Hot Spots HH ({len(hh)})</b><br>
        ADR médian : <b>€{hh['adr'].median():.0f}</b><br>
        Prime de localisation confirmée. Prix d'acquisition élevés.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="insight-box">
        <b>Opportunités HL ({len(hl)})</b><br>
        ADR médian : <b>€{hl['adr'].median():.0f}</b><br>
        Sur-performent dans zones morose → différenciateur caché.
        Cibles d'acquisition prioritaires.
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="alert-box">
        <b>Cold Spots LL ({len(ll)})</b><br>
        ADR médian : <b>€{ll['adr'].median():.0f}</b><br>
        Sous-performance structurelle.
        Évaluer cession avant dégradation.
        </div>""", unsafe_allow_html=True)

    st.subheader("Données détaillées")
    cols_show = [c for c in ["zone","stars","rooms","adr","revpar","lisa"] if c in df.columns]
    st.dataframe(df[cols_show].sort_values("adr", ascending=False),
                 use_container_width=True, height=350)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 : CARTE INTERACTIVE
# ─────────────────────────────────────────────────────────────────────────────
def page_map(df, portfolio):
    st.title("Carte interactive — Marché + Portefeuille")

    col1, col2, col3 = st.columns(3)
    show_heat  = col1.checkbox("Heatmap ADR", value=True)
    show_lisa  = col2.checkbox("Clusters LISA", value=True)
    show_port  = col3.checkbox("Portefeuille", value=True)

    # LISA pour la carte
    if show_lisa and len(df) >= 20:
        lisa = compute_lisa(
            hash(tuple(df["adr"].values[:10])),
            df["adr"].values, df["lat"].values, df["lon"].values
        )
        df = df.copy()
        df["lisa"] = lisa

    center = [df["lat"].mean(), df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=14, tiles=None)
    folium.TileLayer("CartoDB positron", name="Clair",  show=True).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Sombre", show=False).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OSM", show=False).add_to(m)

    if show_heat:
        HeatMap([[r.lat, r.lon, r.adr] for _,r in df.iterrows()],
                name="Heatmap ADR", radius=22, blur=16,
                gradient={0.2:"#313695",0.5:"#ffffbf",1.0:"#a50026"},
                show=True).add_to(m)

    if show_lisa and "lisa" in df.columns:
        for lt in ["HH","LL","HL","LH","NS"]:
            fg = folium.FeatureGroup(name=f"LISA {lt} — {LISA_LABELS[lt][:20]}",
                                      show=(lt!="NS"))
            for _,r in df[df["lisa"]==lt].iterrows():
                folium.CircleMarker(
                    [r.lat, r.lon],
                    radius=4+r.stars,
                    color=LISA_COLORS[lt], fill=True,
                    fill_color=LISA_COLORS[lt],
                    fill_opacity=0.75 if lt!="NS" else 0.2,
                    weight=0.5,
                    tooltip=f"ADR: €{int(r.adr)} | {lt}",
                    popup=f"<b>{r.get('zone','')}</b><br>ADR €{int(r.adr)}<br>{int(r.stars)}★ · {int(r.rooms)} ch."
                ).add_to(fg)
            fg.add_to(m)

    if show_port and len(portfolio) > 0:
        fg_p = folium.FeatureGroup(name="Portefeuille", show=True)
        for _,r in portfolio.iterrows():
            col = DEC_COLORS.get(r.decision, "#888")
            popup_html = f"""
            <div style='font-family:Arial;min-width:180px'>
              <b style='color:{col}'>{r.nom}</b><br>
              {int(r.stars)}★ · {int(r.rooms)} ch.<br>
              ADR : €{int(r.adr)} · RevPAR : €{int(r.revpar)}<br>
              <span style='background:{col};color:white;padding:2px 8px;border-radius:4px;font-size:12px'>
                {r.decision}
              </span>
            </div>"""
            folium.Marker(
                [r.lat, r.lon],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{r.nom} — {r.decision}",
                icon=folium.Icon(color="white", icon_color=col, icon="home", prefix="fa")
            ).add_to(fg_p)
        fg_p.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    MiniMap(toggle_display=True, tile_layer="CartoDB positron").add_to(m)
    st_folium(m, width=None, height=650, returned_objects=[])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 : PORTEFEUILLE
# ─────────────────────────────────────────────────────────────────────────────
def page_portfolio(portfolio):
    st.title("Portefeuille — Matrice de décision")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Valeur totale",  f"€{portfolio['valeur'].sum()}M")
    c2.metric("ADR moyen",      f"€{portfolio['adr'].mean():.0f}")
    c3.metric("RevPAR moyen",   f"€{portfolio['revpar'].mean():.0f}")
    c4.metric("Chambres total", int(portfolio["rooms"].sum()))

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(portfolio, x="revpar", y="valeur",
                         size="rooms", color="decision",
                         color_discrete_map=DEC_COLORS,
                         hover_name="nom", title="RevPAR vs. Valeur (taille = chambres)",
                         labels={"revpar":"RevPAR (€)","valeur":"Valeur (M€)"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        val_by_dec = portfolio.groupby("decision")["valeur"].sum().sort_values(ascending=False)
        fig = px.bar(x=val_by_dec.index, y=val_by_dec.values,
                     color=val_by_dec.index, color_discrete_map=DEC_COLORS,
                     title="Capital par décision (M€)",
                     labels={"x":"Décision","y":"Valeur M€"})
        st.plotly_chart(fig, use_container_width=True)

    # Carte portefeuille
    fig = go.Figure()
    for dec, col in DEC_COLORS.items():
        sub = portfolio[portfolio["decision"]==dec]
        if len(sub)==0: continue
        fig.add_trace(go.Scatter(
            x=sub["lon"], y=sub["lat"], mode="markers+text",
            name=dec,
            marker=dict(size=sub["valeur"]*1.5+12, color=col,
                        line=dict(width=1.5, color="white")),
            text=sub["nom"].str.split().str[0],
            textposition="top center",
            hovertext=sub.apply(lambda r: f"{r.nom}<br>ADR €{int(r.adr)}<br>{r.decision}", axis=1),
            hoverinfo="text"
        ))
    fig.update_layout(title="Carte portefeuille (taille = valeur M€)",
                      xaxis_title="Longitude", yaxis_title="Latitude",
                      height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Détail des actifs")
    disp = portfolio[["nom","stars","rooms","adr","revpar","occ","valeur","decision"]].copy()
    disp["occ"] = (disp["occ"]*100).round(0).astype(int).astype(str)+"%"
    disp.columns = ["Nom","★","Chambres","ADR €","RevPAR €","Occ.","Valeur M€","Décision"]
    st.dataframe(disp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 : SCREENING ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────
def page_screening(df):
    st.title("Screening — Pipeline d'acquisition")

    st.markdown("""
    <div class="info-box">
    <b>Module C3 :</b> Filtre automatique par critères géo-concurrentiels.
    La pénétration ADR compare chaque hôtel à son competitive set local (rayon 1km, même standing ±1★).
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    min_revpar  = col1.slider("RevPAR min (€)", 40, 300, 90)
    min_stars   = col2.slider("Étoiles min",    2, 5,   3)
    max_rooms   = col3.slider("Chambres max",   50, 500, 250)
    min_pen     = col4.slider("Pénétration ADR min", 0.8, 1.3, 0.95, step=0.01,
                               help="Ratio ADR hôtel / ADR competitive set. >1 = dominant")

    # Calcul pénétration ADR
    df = df.copy()
    coords_arr = df[["lon","lat"]].values
    tree = cKDTree(coords_arr)

    pen_scores, n_comps, adr_sets = [], [], []
    for i, row in df.iterrows():
        idx = tree.query_ball_point([row["lon"], row["lat"]], r=0.009)
        comps = df.iloc[idx]
        comps = comps[comps.index != i]
        same  = comps[comps["stars"].between(row["stars"]-1, row["stars"]+1)]
        n_comps.append(len(same))
        adr_set = same["adr"].mean() if len(same) > 0 else np.nan
        adr_sets.append(adr_set)
        pen_scores.append(row["adr"] / adr_set if not np.isnan(adr_set) else np.nan)

    df["n_comp_1km"]     = n_comps
    df["adr_set_avg"]    = adr_sets
    df["penetration_adr"] = pen_scores

    # Filtrage
    targets = df[
        (df["revpar"] >= min_revpar) &
        (df["stars"]  >= min_stars) &
        (df["rooms"]  <= max_rooms) &
        (df["penetration_adr"].notna()) &
        (df["penetration_adr"] >= min_pen)
    ].copy().sort_values("penetration_adr", ascending=False)

    st.metric("Cibles identifiées", len(targets),
              delta=f"{len(targets)/len(df)*100:.1f}% du marché analysé")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["lon"], y=df["lat"], mode="markers",
                                  name="Marché", marker=dict(size=4, color="#CCCCCC", opacity=0.4)))
        if len(targets):
            fig.add_trace(go.Scatter(
                x=targets["lon"], y=targets["lat"], mode="markers",
                name="Cibles",
                marker=dict(size=10, color=targets["penetration_adr"],
                            colorscale="RdYlGn", showscale=True,
                            colorbar=dict(title="Pénétration ADR"),
                            line=dict(width=1, color="white")),
                hovertext=targets.apply(
                    lambda r: f"ADR €{int(r.adr)} vs set €{int(r.adr_set_avg):.0f}", axis=1),
                hoverinfo="text"
            ))
        fig.update_layout(title="Carte des cibles d'acquisition",
                          xaxis_title="Longitude", yaxis_title="Latitude")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if len(targets) > 0:
            fig = px.scatter(targets, x="adr_set_avg", y="adr",
                             color="penetration_adr", color_continuous_scale="RdYlGn",
                             size="rooms", size_max=20,
                             title="ADR hôtel vs. competitive set",
                             labels={"adr_set_avg":"ADR competitive set (€)","adr":"ADR hôtel (€)"})
            max_v = max(targets["adr"].max(), targets["adr_set_avg"].max()) + 20
            fig.add_shape(type="line", x0=40,y0=40,x1=max_v,y1=max_v,
                          line=dict(dash="dash",color="black",width=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune cible avec ces critères.")

    if len(targets) > 0:
        st.subheader(f"Top cibles ({min(15,len(targets))})")
        cols_show = [c for c in ["zone","stars","rooms","adr","revpar",
                                  "adr_set_avg","penetration_adr","n_comp_1km"] if c in targets.columns]
        disp = targets[cols_show].head(15).copy()
        disp["penetration_adr"] = disp["penetration_adr"].round(3)
        disp["adr_set_avg"]     = disp["adr_set_avg"].round(0)
        st.dataframe(disp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 : EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def page_export(df, portfolio):
    st.title("Export — Insights client")

    st.markdown("""
    <div class="info-box">
    Téléchargez les données enrichies (LISA + GWR scores + pénétration ADR)
    et le résumé des insights pour intégration dans vos rapports client.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Résumé exécutif")

    # Insights automatiques
    st.markdown(f"""
    **Marché analysé :** {len(df)} hôtels · ADR moyen **€{df['adr'].mean():.0f}** · RevPAR moyen **€{df['revpar'].mean():.0f}**

    **Portefeuille :** {len(portfolio)} actifs · Valeur totale **€{portfolio['valeur'].sum()}M**
    - HOLD : {(portfolio['decision']=='HOLD ★').sum()} actifs (€{portfolio[portfolio['decision']=='HOLD ★']['valeur'].sum()}M)
    - CAPEX : {(portfolio['decision']=='CAPEX').sum()} actifs
    - À céder : {(portfolio['decision']=='CÉDER').sum()} actifs (€{portfolio[portfolio['decision']=='CÉDER']['valeur'].sum()}M libérés)
    """)

    st.divider()
    st.subheader("Téléchargements")

    col1, col2, col3 = st.columns(3)

    # Export marché
    with col1:
        csv_market = df.to_csv(index=False).encode("utf-8")
        st.download_button("Données marché (CSV)",
                           data=csv_market,
                           file_name="hotel_market_data.csv",
                           mime="text/csv")

    # Export portefeuille
    with col2:
        csv_port = portfolio.to_csv(index=False).encode("utf-8")
        st.download_button("Portefeuille (CSV)",
                           data=csv_port,
                           file_name="hotel_portfolio.csv",
                           mime="text/csv")

    # Export résumé texte
    with col3:
        summary_txt = f"""HOTEL GEO-ANALYTICS — RAPPORT SYNTHÈSE
{'='*50}

MARCHÉ
Hôtels analysés  : {len(df)}
ADR moyen        : €{df['adr'].mean():.0f}
RevPAR moyen     : €{df['revpar'].mean():.0f}
Occupation moy.  : {df['occ'].mean()*100:.0f}%

PORTEFEUILLE
Actifs           : {len(portfolio)}
Valeur totale    : €{portfolio['valeur'].sum()}M
ADR moyen        : €{portfolio['adr'].mean():.0f}
RevPAR moyen     : €{portfolio['revpar'].mean():.0f}

DÉCISIONS
HOLD             : {(portfolio['decision']=='HOLD ★').sum()} actifs
CAPEX            : {(portfolio['decision']=='CAPEX').sum()} actifs
REPOSITIONNER    : {(portfolio['decision']=='REPOSITIONNER').sum()} actifs
CÉDER            : {(portfolio['decision']=='CÉDER').sum()} actifs
SURVEILLER       : {(portfolio['decision']=='SURVEILLER').sum()} actifs
"""
        st.download_button("Résumé exécutif (TXT)",
                           data=summary_txt.encode("utf-8"),
                           file_name="hotel_geoanalytics_summary.txt",
                           mime="text/plain")

    # Template CSV pour upload
    st.divider()
    st.subheader("Templates CSV pour vos données réelles")
    col1, col2 = st.columns(2)

    with col1:
        template_market = pd.DataFrame({
            "lat":[48.860,48.855],"lon":[2.350,2.341],
            "adr":[195,175],"stars":[4,3],"rooms":[120,80],
            "occ":[0.75,0.70],"revpar":[146,122],"zone":["Centre","Historique"]
        })
        st.download_button("Template marché CSV",
                           data=template_market.to_csv(index=False).encode("utf-8"),
                           file_name="template_market.csv", mime="text/csv")
        st.caption("Colonnes requises : lat, lon, adr, stars, rooms")

    with col2:
        template_port = pd.DataFrame({
            "nom":["Hotel A","Hotel B"],"lat":[48.860,48.855],
            "lon":[2.350,2.341],"stars":[5,4],"rooms":[180,120],
            "adr":[310,210],"occ":[0.77,0.77],"decision":["HOLD ★","CAPEX"],"valeur":[45,28]
        })
        st.download_button("Template portefeuille CSV",
                           data=template_port.to_csv(index=False).encode("utf-8"),
                           file_name="template_portfolio.csv", mime="text/csv")
        st.caption("Colonnes requises : nom, lat, lon, adr, stars, rooms, decision")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    page, df, portfolio = sidebar()

    if   page == "📊 Dashboard":                   page_dashboard(df, portfolio)
    elif page == "📍 GWR + Scoring ADR manquant":  page_gwr(df)
    elif page == "🔗 LISA Clustering":             page_lisa(df)
    elif page == "🗺️ Carte interactive":           page_map(df, portfolio)
    elif page == "💼 Portefeuille":                page_portfolio(portfolio)
    elif page == "🎯 Screening acquisition":       page_screening(df)
    elif page == "📥 Export insights":             page_export(df, portfolio)

if __name__ == "__main__":
    main()
