import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Teach the Machine",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp { background: #f5f5f5; }
.block-container { padding: 2rem 2.5rem 1rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] input {
    background: #252542 !important;
    border: 1px solid #3a3a5c !important;
    color: white !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stNumberInput input {
    background: #252542 !important;
    color: white !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #EB0000 !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888 !important;
}

/* Buttons */
.stButton > button {
    background: #EB0000 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.2rem !important;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #c40000 !important; }

/* Cards */
.card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.75rem;
}

/* Leaderboard entries */
.lb-entry {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 12px;
    background: #f9f9f9;
    margin-bottom: 8px;
    border: 1px solid #f0f0f0;
}
.lb-rank { font-size: 1.4rem; min-width: 32px; }
.lb-name { font-weight: 600; font-size: 0.95rem; flex: 1; color: #1a1a2e; }
.lb-score { font-size: 1.3rem; font-weight: 800; min-width: 48px; text-align: right; }
.lb-bar-bg { background: #f0f0f0; height: 6px; border-radius: 99px; margin-top: 5px; }
.lb-weights { font-size: 0.72rem; color: #9ca3af; margin-top: 3px; font-family: monospace; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Number inputs in sidebar */
[data-testid="stSidebar"] .stNumberInput > div > div {
    background: #252542 !important;
    border-radius: 8px !important;
}

/* Accuracy accent */
.acc-big {
    font-size: 3rem;
    font-weight: 800;
    color: #EB0000;
    line-height: 1;
}
.acc-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #888;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── DATA ────────────────────────────────────────────────
TRAIN = [(0,0,0),(7,1,1),(4,1,1),(1,0,0),(5,0,1),(3,1,1),(2,0,0),(6,1,1)]
TESTFALL = (3, 0)
SHEET_ID = "1EFjFomoYYxq9q2tQW0EenEj3KUDaPvDGn0h2Fo8jKxY"

def calc_accuracy(w1, w2, bias):
    return sum(1 for x1,x2,y in TRAIN if (1 if w1*x1+w2*x2+bias>0 else 0)==y)

@st.cache_resource
def get_worksheet():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    )
    ws = gspread.authorize(creds).open_by_key(SHEET_ID).sheet1
    if ws.acell("A1").value != "Name":
        ws.update("A1:E1",[["Name","w1","w2","bias","Accuracy"]])
    return ws

@st.cache_data(ttl=8)
def load_entries():
    return get_worksheet().get_all_records()

def save_entry(name, w1, w2, bias, acc):
    ws = get_worksheet()
    rows = ws.get_all_records()
    for i,r in enumerate(rows):
        if str(r.get("Name","")).strip().lower()==name.strip().lower():
            ws.update(f"A{i+2}:E{i+2}",[[name,w1,w2,bias,acc]])
            st.cache_data.clear(); return
    ws.append_row([name,w1,w2,bias,acc])
    st.cache_data.clear()

def delete_entry(name):
    ws = get_worksheet()
    rows = ws.get_all_records()
    for i,r in enumerate(rows):
        if str(r.get("Name","")).strip()==name.strip():
            ws.delete_rows(i+2); st.cache_data.clear(); return

def make_plot(w1, w2, bias):
    fig = go.Figure()
    x1r = np.linspace(-0.5, 8.5, 300)
    if abs(w2) > 0.001:
        fig.add_trace(go.Scatter(
            x=x1r, y=-(w1*x1r+bias)/w2,
            mode='lines', line=dict(color='#EB0000',width=2.5,dash='dash'),
            name='Entscheidungsgrenze', showlegend=True
        ))
    for label,color,symbol,lname in [(0,'#1a1a2e','circle','Pünktlich'),(1,'#EB0000','square','Verspätung')]:
        pts=[(x1,x2) for x1,x2,y in TRAIN if y==label]
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts],
                mode='markers', name=lname,
                marker=dict(size=14,color=color,symbol=symbol,line=dict(width=2,color='white'))
            ))
    fig.add_trace(go.Scatter(
        x=[TESTFALL[0]],y=[TESTFALL[1]], mode='markers', name='Testfall',
        marker=dict(size=16,color='#f59e0b',symbol='star',line=dict(width=2,color='white'))
    ))
    fig.update_layout(
        xaxis_title='Verspätung Vorstation (Min.)',
        yaxis_title='Rushhour',
        xaxis=dict(range=[-0.5,8.5],tickvals=list(range(9)),showgrid=True,gridcolor='#f0f0f0'),
        yaxis=dict(range=[-0.3,1.3],tickvals=[0,1],ticktext=['Nein','Ja'],showgrid=True,gridcolor='#f0f0f0'),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0,font=dict(size=12)),
        margin=dict(l=50,r=20,t=40,b=50), height=380,
        font=dict(family='Inter, sans-serif',size=12),
        hoverlabel=dict(bgcolor='white',font_size=13)
    )
    return fig

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem;'>
        <div style='font-size:1.4rem;font-weight:800;color:white;'>Teach the Machine</div>
        <div style='font-size:0.75rem;color:#666;margin-top:4px;letter-spacing:.06em;text-transform:uppercase;'>Workshop · SBB</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:#666;margin-bottom:.5rem;'>Dein Name</div>", unsafe_allow_html=True)
    name = st.text_input("", placeholder="z.B. Ilir F.", label_visibility="collapsed")

    st.markdown("<div style='margin:1rem 0 .5rem;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:#666;'>Gewichte einstellen</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.75rem;color:#555;margin-bottom:.75rem;font-family:monospace;'>z = w1·Versp + w2·Rush + bias</div>", unsafe_allow_html=True)

    w1   = st.number_input("w1 — Verspätung", value=0.0, step=0.1, format="%.2f")
    w2   = st.number_input("w2 — Rushhour",   value=0.0, step=0.1, format="%.2f")
    bias = st.number_input("Bias",             value=0.0, step=0.1, format="%.2f")

    acc = calc_accuracy(w1, w2, bias)
    pct = int(acc/8*100)
    color = "#22c55e" if acc==8 else "#EB0000" if acc<4 else "#f59e0b"

    st.markdown(f"""
    <div style='margin:1.25rem 0;padding:1.25rem;background:#252542;border-radius:12px;text-align:center;'>
        <div style='font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;color:#666;margin-bottom:.3rem;'>Accuracy</div>
        <div style='font-size:2.8rem;font-weight:800;color:{color};line-height:1;'>{acc}/8</div>
        <div style='font-size:.8rem;color:#666;margin-top:.3rem;'>{pct}%</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📤 Ins Leaderboard eintragen"):
        if not name.strip():
            st.error("Bitte Namen eingeben.")
        else:
            with st.spinner(""):
                save_entry(name.strip(), round(w1,2), round(w2,2), round(bias,2), acc)
            st.success("✓ Eingetragen!")
            st.rerun()

    st.markdown("""
    <div style='margin-top:2rem;padding-top:1.5rem;border-top:1px solid #2a2a4a;'>
        <div style='font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;color:#555;margin-bottom:.75rem;'>Trainingsdaten</div>
    </div>
    """, unsafe_allow_html=True)

    data_rows = ""
    for i,(x1,x2,y) in enumerate(TRAIN):
        dot = "🟢" if y==0 else "🔴"
        rush = "Ja" if x2==1 else "Nein"
        label = "Pünktlich" if y==0 else "Verspätung"
        data_rows += f"<tr><td style='padding:4px 6px;color:#888;'>{i+1}</td><td style='padding:4px 6px;color:#ccc;'>{x1} Min.</td><td style='padding:4px 6px;color:#ccc;'>{rush}</td><td style='padding:4px 6px;'>{dot} {label}</td></tr>"

    st.markdown(f"""
    <table style='width:100%;font-size:.78rem;border-collapse:collapse;'>
        <thead><tr>
            <th style='padding:4px 6px;color:#555;font-weight:600;text-align:left;'>#</th>
            <th style='padding:4px 6px;color:#555;font-weight:600;text-align:left;'>Versp.</th>
            <th style='padding:4px 6px;color:#555;font-weight:600;text-align:left;'>Rush</th>
            <th style='padding:4px 6px;color:#555;font-weight:600;text-align:left;'>Label</th>
        </tr></thead>
        <tbody>{data_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────
col1, col2 = st.columns([1.3, 1], gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-title">Entscheidungsgrenze</div>', unsafe_allow_html=True)
    st.plotly_chart(make_plot(w1, w2, bias), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔒 Testfall aufdecken — erst nach dem Hands-on!"):
        z = w1*TESTFALL[0] + w2*TESTFALL[1] + bias
        pred = "🔴 Verspätung" if z > 0 else "✅ Pünktlich"
        correct = "✅ Pünktlich"
        st.markdown(f"""
        **Testfall:** 3 Min. Verspätung, kein Rushhour

        `z = {w1:.2f} × 3 + {w2:.2f} × 0 + ({bias:.2f}) = {z:.3f}`

        **Vorhersage:** {pred} &nbsp;|&nbsp; **Korrekt:** {correct}
        """)

with col2:
    st.markdown('<div class="card"><div class="card-title">🏆 Rangliste</div>', unsafe_allow_html=True)
    try:
        entries = load_entries()
        if not entries:
            st.markdown("<div style='color:#9ca3af;font-size:.9rem;padding:.5rem 0;'>Noch keine Einträge.</div>", unsafe_allow_html=True)
        else:
            entries_sorted = sorted(entries, key=lambda x: -int(x.get('Accuracy',0)))
            medals = ['🥇','🥈','🥉']

            for i, e in enumerate(entries_sorted):
                a = int(e.get('Accuracy',0))
                medal = medals[i] if i < 3 else f"<span style='font-size:.9rem;color:#9ca3af;font-weight:700;'>#{i+1}</span>"
                bar_color = '#22c55e' if a==8 else '#EB0000' if a<4 else '#f59e0b'
                bar_w = int(a/8*100)
                col_e, col_x = st.columns([12,1])
                with col_e:
                    st.markdown(f"""
                    <div style='padding:12px 14px;background:#fafafa;border:1px solid #f0f0f0;border-radius:12px;margin-bottom:6px;'>
                        <div style='display:flex;align-items:center;justify-content:space-between;'>
                            <div style='display:flex;align-items:center;gap:10px;'>
                                <span style='font-size:1.3rem;'>{medal if i<3 else ''}</span>
                                {"" if i<3 else f"<span style='font-size:.85rem;color:#9ca3af;font-weight:700;min-width:24px;'>#{i+1}</span>"}
                                <span style='font-weight:700;font-size:.95rem;color:#1a1a2e;'>{e.get('Name','—')}</span>
                            </div>
                            <span style='font-size:1.3rem;font-weight:800;color:{bar_color};'>{a}/8</span>
                        </div>
                        <div style='background:#f0f0f0;height:5px;border-radius:99px;margin-top:8px;overflow:hidden;'>
                            <div style='width:{bar_w}%;height:100%;background:{bar_color};border-radius:99px;'></div>
                        </div>
                        <div style='font-size:.7rem;color:#9ca3af;margin-top:5px;font-family:monospace;'>w1={e.get('w1','?')} · w2={e.get('w2','?')} · bias={e.get('bias','?')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_x:
                    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
                    if st.button("✕", key=f"del_{i}_{e.get('Name','')}", help="Löschen"):
                        delete_entry(str(e.get('Name','')))
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("🔄 Aktualisieren"):
            st.cache_data.clear(); st.rerun()

    except Exception as ex:
        st.error(f"Fehler: {ex}")
        st.markdown('</div>', unsafe_allow_html=True)
