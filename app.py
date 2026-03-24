import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
import numpy as np
import json
import time

# ── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="Teach the Machine — Leaderboard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLES ───────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #ffffff; }
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
  h1 { color: #EB0000; font-size: 2rem !important; }
  .metric-box { background:#f8f8f8; border:1px solid #e0e0e0; padding:16px 20px; border-radius:0; }
  .rank-1 { color: #EB0000; font-weight: 800; font-size: 1.4rem; }
  .rank-2 { color: #1a3f6f; font-weight: 700; }
  .rank-3 { color: #374151; font-weight: 600; }
  div[data-testid="stForm"] { border: none; padding: 0; }
  .stButton>button { background:#EB0000; color:white; border:none; border-radius:0; font-weight:700; padding:10px 24px; width:100%; }
  .stButton>button:hover { background:#c0000d; color:white; }
  .stNumberInput input { border-radius:0; }
  .stTextInput input { border-radius:0; }
</style>
""", unsafe_allow_html=True)

# ── TRAINING DATA ────────────────────────────────────────
TRAIN = [
    (0, 0, 0), (7, 1, 1), (4, 1, 1), (1, 0, 0),
    (5, 0, 1), (3, 1, 1), (2, 0, 0), (6, 1, 1),
]  # (verspätung_min, rushhour, label)

TESTFALL = (3, 0)  # → pünktlich (z = 0.5*3 + 1.5*0 - 2.2 = -0.7)

# ── GOOGLE SHEETS ────────────────────────────────────────
SHEET_ID = "1EFjFomoYYxq9q2tQW0EenEj3KUDaPvDGn0h2Fo8jKxY"

@st.cache_resource
def get_worksheet():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.sheet1
    if ws.acell("A1").value != "Name":
        ws.update("A1:E1", [["Name","w1","w2","bias","Accuracy"]])
    return ws

@st.cache_data(ttl=8)
def load_entries():
    ws = get_worksheet()
    return ws.get_all_records()

def save_entry(name, w1, w2, bias, acc):
    ws = get_worksheet()
    rows = ws.get_all_records()
    for i, r in enumerate(rows):
        if str(r.get("Name","")).strip().lower() == name.strip().lower():
            ws.update(f"A{i+2}:E{i+2}", [[name, w1, w2, bias, acc]])
            st.cache_data.clear()
            return
    ws.append_row([name, w1, w2, bias, acc])
    st.cache_data.clear()

# ── ACCURACY CALCULATION ─────────────────────────────────
def calc_accuracy(w1, w2, bias):
    correct = 0
    for x1, x2, y in TRAIN:
        z = w1 * x1 + w2 * x2 + bias
        pred = 1 if z > 0 else 0
        if pred == y:
            correct += 1
    return correct

def predict(w1, w2, bias, x1, x2):
    z = w1 * x1 + w2 * x2 + bias
    return 1 if z > 0 else 0, z

# ── DECISION BOUNDARY PLOT ───────────────────────────────
def make_plot(w1, w2, bias):
    fig = go.Figure()

    # Decision boundary: w1*x1 + w2*x2 + bias = 0 → x2 = -(w1*x1 + bias)/w2
    x1_range = np.linspace(-0.5, 8.5, 200)

    if abs(w2) > 0.001:
        x2_line = -(w1 * x1_range + bias) / w2
        fig.add_trace(go.Scatter(
            x=x1_range, y=x2_line,
            mode='lines',
            line=dict(color='#EB0000', width=2, dash='dash'),
            name='Entscheidungsgrenze'
        ))

    # Training points
    colors_map = {0: '#1a3f6f', 1: '#EB0000'}
    symbols_map = {0: 'circle', 1: 'square'}
    labels_map = {0: 'Pünktlich', 1: 'Verspätung'}

    for label in [0, 1]:
        pts = [(x1, x2) for x1, x2, y in TRAIN if y == label]
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode='markers',
                marker=dict(size=14, color=colors_map[label],
                           symbol=symbols_map[label], line=dict(width=1.5, color='white')),
                name=labels_map[label]
            ))

    # Testfall
    fig.add_trace(go.Scatter(
        x=[TESTFALL[0]], y=[TESTFALL[1]],
        mode='markers',
        marker=dict(size=16, color='#f59e0b', symbol='star',
                   line=dict(width=2, color='white')),
        name='Testfall (3 Min., kein Rushhour)'
    ))

    fig.update_layout(
        xaxis_title='Verspätung Vorstation (Min.)',
        yaxis_title='Rushhour (0=Nein, 1=Ja)',
        xaxis=dict(range=[-0.5, 8.5], tickvals=list(range(9))),
        yaxis=dict(range=[-0.3, 1.3], tickvals=[0, 1], ticktext=['Nein', 'Ja']),
        plot_bgcolor='#f8f8f8',
        paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=40, r=20, t=40, b=40),
        height=360,
        font=dict(family='sans-serif', size=13)
    )
    return fig

# ── SIDEBAR: EINGABE ─────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎚️ Deine Gewichte")
    st.markdown("Formel: `z = w1 × Verspätung + w2 × Rushhour + bias`")
    st.markdown("`z > 0` → Verspätung &nbsp;|&nbsp; `z ≤ 0` → Pünktlich")
    st.markdown("---")

    name = st.text_input("Dein Name", placeholder="z.B. Ilir F.")
    w1   = st.number_input("w1 — Verspätung", value=0.0, step=0.1, format="%.2f")
    w2   = st.number_input("w2 — Rushhour",   value=0.0, step=0.1, format="%.2f")
    bias = st.number_input("Bias",             value=0.0, step=0.1, format="%.2f")

    acc = calc_accuracy(w1, w2, bias)
    st.markdown("---")
    st.metric("Accuracy", f"{acc} / 8", delta=f"{int(acc/8*100)}%")

    submit = st.button("📤 Ins Leaderboard eintragen")

    if submit:
        if not name.strip():
            st.error("Bitte Namen eingeben.")
        else:
            try:
                with st.spinner("Wird gespeichert..."):
                    save_entry(name.strip(), round(w1,2), round(w2,2), round(bias,2), acc)
                st.success("✓ Eingetragen!")
                st.rerun()
            except Exception as e:
                st.error(f"Fehler: {e}")

    st.markdown("---")
    st.markdown("**Trainings-Datensatz**")
    st.markdown("""
| # | Versp. | Rush | Label |
|---|--------|------|-------|
| 1 | 0 Min  | Nein | ✅ Pünktlich |
| 2 | 7 Min  | Ja   | 🔴 Verspätung |
| 3 | 4 Min  | Ja   | 🔴 Verspätung |
| 4 | 1 Min  | Nein | ✅ Pünktlich |
| 5 | 5 Min  | Nein | 🔴 Verspätung |
| 6 | 3 Min  | Ja   | 🔴 Verspätung |
| 7 | 2 Min  | Nein | ✅ Pünktlich |
| 8 | 6 Min  | Ja   | 🔴 Verspätung |
""")

# ── MAIN AREA ─────────────────────────────────────────────
st.markdown("# 🚂 Teach the Machine — Leaderboard")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📊 Entscheidungsgrenze")
    st.plotly_chart(make_plot(w1, w2, bias), use_container_width=True)

    # Testfall
    with st.expander("🔒 Testfall aufdecken (erst nach dem Hands-on!)"):
        pred, z = predict(w1, w2, bias, TESTFALL[0], TESTFALL[1])
        st.markdown(f"""
**Testfall:** 3 Min. Verspätung, kein Rushhour

`z = {w1:.2f} × 3 + {w2:.2f} × 0 + {bias:.2f} = {z:.2f}`

**Vorhersage:** {'🔴 Verspätung' if pred==1 else '✅ Pünktlich'}

**Korrekte Antwort:** ✅ Pünktlich (z sollte < 0 sein)
""")

with col2:
    st.markdown("### 🏆 Rangliste")

    try:
        entries = load_entries()

        if not entries:
            st.info("Noch keine Einträge. Sei der Erste!")
        else:
            entries_sorted = sorted(entries, key=lambda x: -int(x.get('Accuracy', 0)))
            medals = ['🥇', '🥈', '🥉']
            for i, e in enumerate(entries_sorted):
                acc_val = int(e.get('Accuracy', 0))
                medal = medals[i] if i < 3 else f"#{i+1}"
                bar_color = '#EB0000' if acc_val == 8 else '#1a3f6f' if acc_val >= 6 else '#9ca3af'
                bar_width = int(acc_val / 8 * 100)
                col_entry, col_del = st.columns([11, 1])
                with col_entry:
                    st.markdown(f"""
<div style="padding:10px 14px;background:#f8f8f8;border:1px solid #e0e0e0;margin-bottom:6px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="font-size:15px;font-weight:700;">{medal} {e.get('Name','—')}</span>
    <span style="font-size:18px;font-weight:800;color:{bar_color};">{acc_val}/8</span>
  </div>
  <div style="background:#e0e0e0;height:8px;">
    <div style="width:{bar_width}%;height:100%;background:{bar_color};"></div>
  </div>
  <div style="font-size:11px;color:#6b7280;margin-top:4px;">
    w1={e.get('w1','?')} &nbsp;·&nbsp; w2={e.get('w2','?')} &nbsp;·&nbsp; bias={e.get('bias','?')}
  </div>
</div>
""", unsafe_allow_html=True)
                with col_del:
                    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
                    if st.button("✕", key=f"del_{i}_{e.get('Name','')}", help=f"{e.get('Name','')} löschen"):
                        try:
                            ws = get_worksheet()
                            all_rows = ws.get_all_records()
                            for j, r in enumerate(all_rows):
                                if str(r.get("Name","")).strip() == str(e.get("Name","")).strip():
                                    ws.delete_rows(j + 2)
                                    break
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Fehler: {ex}")

        if st.button("🔄 Aktualisieren"):
            st.cache_data.clear()
            st.rerun()

    except Exception as e:
        st.error(f"Google Sheets Fehler: {e}")
        st.info("Stelle sicher dass das Sheet mit dem Service Account geteilt ist.")
