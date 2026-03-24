import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import plotly.graph_objects as go
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Teach the Machine",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container { padding: 2rem 2.5rem 1rem; }
[data-testid="stSidebar"] { background: #1a1a2e; }
[data-testid="stSidebar"] label { color: #aaa !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stSidebar"] input { background: #252542 !important; color: white !important; border: 1px solid #3a3a5c !important; border-radius: 8px !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div { color: #ccc; }
.stButton > button { border-radius: 8px !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ── TRAINING DATA (8 Züge, linear trennbar) ─────────────
TRAIN = [(0,0,0),(7,1,1),(4,1,1),(1,0,0),(5,0,1),(3,1,1),(2,0,0),(6,1,1)]
TESTFALL = (3, 0)
SHEET_ID = "1EFjFomoYYxq9q2tQW0EenEj3KUDaPvDGn0h2Fo8jKxY"

# ── COMPLEX DATA (für Overfitting-Tab) ──────────────────
# Nicht-lineares Muster: mittlere Verspätung + kein Rushhour = manchmal pünktlich wegen Aufholen
np.random.seed(7)
def make_complex_data(n=60, noise=True):
    pts = []
    for _ in range(n):
        x1 = np.random.uniform(0, 10)
        x2 = np.random.choice([0, 1])
        # Nicht-lineare Regel: Verspätung + Rushhour * 2 + sin(x1)*Rushhour > 4
        signal = x1 * 0.5 + x2 * 1.8 + np.sin(x1) * x2 * 0.8
        if noise:
            signal += np.random.normal(0, 0.5)
        y = 1 if signal > 2.5 else 0
        pts.append((x1, x2, y))
    return pts

COMPLEX_ALL = make_complex_data(80)
COMPLEX_TRAIN = COMPLEX_ALL[:50]
COMPLEX_TEST  = COMPLEX_ALL[50:]

def calc_accuracy(w1, w2, bias, data=None):
    if data is None: data = TRAIN
    return sum(1 for x1,x2,y in data if (1 if w1*x1+w2*x2+bias>0 else 0)==y)

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
    for i,r in enumerate(ws.get_all_records()):
        if str(r.get("Name","")).strip()==name.strip():
            ws.delete_rows(i+2); st.cache_data.clear(); return

def make_boundary_plot(w1, w2, bias):
    fig = go.Figure()
    x1r = np.linspace(-0.5, 8.5, 300)
    if abs(w2) > 0.001:
        fig.add_trace(go.Scatter(
            x=x1r, y=-(w1*x1r+bias)/w2, mode='lines',
            line=dict(color='#EB0000',width=2.5,dash='dash'),
            name='Entscheidungsgrenze'
        ))
    for label,color,symbol,lname in [(0,'#1a1a2e','circle','Pünktlich'),(1,'#EB0000','square','Verspätung')]:
        pts = [(x1,x2) for x1,x2,y in TRAIN if y==label]
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts],
                mode='markers', name=lname,
                marker=dict(size=14,color=color,symbol=symbol,line=dict(width=2,color='white'))
            ))
    fig.add_trace(go.Scatter(
        x=[TESTFALL[0]], y=[TESTFALL[1]], mode='markers', name='Testfall',
        marker=dict(size=16,color='#f59e0b',symbol='star',line=dict(width=2,color='white'))
    ))
    fig.update_layout(
        xaxis_title='Verspätung Vorstation (Min.)',
        yaxis_title='Rushhour',
        xaxis=dict(range=[-0.5,8.5],tickvals=list(range(9)),gridcolor='#f0f0f0'),
        yaxis=dict(range=[-0.3,1.3],tickvals=[0,1],ticktext=['Nein','Ja'],gridcolor='#f0f0f0'),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h',yanchor='bottom',y=1.02),
        margin=dict(l=50,r=20,t=40,b=50), height=380,
        font=dict(size=12)
    )
    return fig

def make_fitting_plot(complexity, show_test):
    X_tr = np.array([(x1,x2) for x1,x2,y in COMPLEX_TRAIN])
    y_tr = np.array([y for x1,x2,y in COMPLEX_TRAIN])
    X_te = np.array([(x1,x2) for x1,x2,y in COMPLEX_TEST])
    y_te = np.array([y for x1,x2,y in COMPLEX_TEST])

    # Train model with varying complexity via SVM RBF gamma
    if complexity == 1:
        model = Pipeline([('poly', PolynomialFeatures(1)), ('lr', LogisticRegression(C=0.1, max_iter=1000))])
        label = "Gerade (linear) — zu simpel"
        color = "#1a3f6f"
    elif complexity <= 4:
        model = Pipeline([('poly', PolynomialFeatures(complexity)), ('lr', LogisticRegression(C=1.0, max_iter=1000))])
        label = f"Polynomgrad {complexity} — gute Balance"
        color = "#16a34a"
    else:
        gamma_map = {5:1, 6:3, 7:8, 8:20, 9:50, 10:200}
        g = gamma_map.get(complexity, 200)
        model = SVC(kernel='rbf', gamma=g, C=100, probability=True)
        label = f"Sehr komplex (γ={g}) — Overfitting"
        color = "#EB0000"

    model.fit(X_tr, y_tr)
    train_acc = model.score(X_tr, y_tr)
    test_acc  = model.score(X_te, y_te)

    # Decision boundary mesh
    xx, yy = np.meshgrid(np.linspace(-0.5,10.5,120), np.linspace(-0.3,1.3,60))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig = go.Figure()

    # Background regions
    fig.add_trace(go.Contour(
        x=np.linspace(-0.5,10.5,120), y=np.linspace(-0.3,1.3,60), z=Z,
        showscale=False, colorscale=[[0,'rgba(26,63,111,0.08)'],[1,'rgba(235,0,0,0.08)']],
        contours=dict(coloring='fill', showlines=False), hoverinfo='skip'
    ))

    # Boundary line
    fig.add_trace(go.Contour(
        x=np.linspace(-0.5,10.5,120), y=np.linspace(-0.3,1.3,60), z=Z,
        showscale=False, colorscale=[[0,color],[1,color]],
        contours=dict(coloring='lines', showlines=True, start=0.5, end=0.5, size=0),
        line=dict(width=2.5, dash='dash'), hoverinfo='skip',
        name='Entscheidungsgrenze'
    ))

    # Training points
    for lbl,clr,sym,nm in [(0,'#1a1a2e','circle','Pünktlich (Training)'),(1,'#EB0000','square','Verspätung (Training)')]:
        pts = [(x1,x2) for x1,x2,y in COMPLEX_TRAIN if y==lbl]
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts],
                mode='markers', name=nm,
                marker=dict(size=10,color=clr,symbol=sym,line=dict(width=1.5,color='white'))
            ))

    # Test points
    if show_test:
        for lbl,clr,sym,nm in [(0,'#93c5fd','circle-open','Pünktlich (Test)'),(1,'#fca5a5','square-open','Verspätung (Test)')]:
            pts = [(x1,x2) for x1,x2,y in COMPLEX_TEST if y==lbl]
            if pts:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in pts], y=[p[1] for p in pts],
                    mode='markers', name=nm,
                    marker=dict(size=10,color=clr,symbol=sym,line=dict(width=2,color=clr))
                ))

    fig.update_layout(
        xaxis_title='Verspätung Vorstation (Min.)',
        yaxis_title='Rushhour',
        xaxis=dict(range=[-0.5,10.5],gridcolor='#f0f0f0'),
        yaxis=dict(range=[-0.3,1.3],tickvals=[0,1],ticktext=['Nein','Ja'],gridcolor='#f0f0f0'),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h',yanchor='bottom',y=1.02,font=dict(size=11)),
        margin=dict(l=50,r=20,t=40,b=50), height=420,
        font=dict(size=12),
        title=dict(text=label, font=dict(color=color, size=14))
    )
    return fig, train_acc, test_acc

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚂 Teach the Machine")
    st.caption("Workshop · SBB TechSkill")
    st.divider()
    name = st.text_input("Dein Name", placeholder="z.B. Ilir F.")
    st.caption("Formel: z = w1·Versp + w2·Rush + bias")
    w1   = st.number_input("w1 — Verspätung", value=0.0, step=0.1, format="%.2f")
    w2   = st.number_input("w2 — Rushhour",   value=0.0, step=0.1, format="%.2f")
    bias = st.number_input("Bias",             value=0.0, step=0.1, format="%.2f")
    acc = calc_accuracy(w1, w2, bias)
    st.divider()
    st.metric("Accuracy", f"{acc} / 8", f"{int(acc/8*100)}%")
    if st.button("📤 Ins Leaderboard eintragen", type="primary", use_container_width=True):
        if not name.strip():
            st.error("Bitte Namen eingeben.")
        else:
            with st.spinner("Speichern..."):
                save_entry(name.strip(), round(w1,2), round(w2,2), round(bias,2), acc)
            st.success("✓ Eingetragen!")
            st.rerun()
    st.divider()
    st.caption("**Trainingsdaten**")
    for i,(x1,x2,y) in enumerate(TRAIN):
        icon = "✅" if y==0 else "🔴"
        st.caption(f"{i+1}. {x1} Min. · {'Ja' if x2 else 'Nein'} → {icon}")

# ── TABS ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Leaderboard", "🧠 Was lernt die KI?", "📉 Overfitting"])

# ════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.3, 1], gap="large")

    with col1:
        st.subheader("Entscheidungsgrenze")
        st.plotly_chart(make_boundary_plot(w1,w2,bias), use_container_width=True, config={"displayModeBar":False})
        with st.expander("🔒 Testfall aufdecken — erst nach dem Hands-on!"):
            z = w1*TESTFALL[0] + w2*TESTFALL[1] + bias
            pred = "🔴 Verspätung" if z > 0 else "✅ Pünktlich"
            st.markdown(f"""
**Testfall:** 3 Min. Verspätung, kein Rushhour

`z = {w1:.2f} × 3 + {w2:.2f} × 0 + ({bias:.2f}) = {z:.3f}`

**Vorhersage deines Modells:** {pred}

**Korrekte Antwort:** ✅ Pünktlich
            """)

    with col2:
        st.subheader("🏆 Rangliste")
        if st.button("🔄 Aktualisieren", use_container_width=True):
            st.cache_data.clear(); st.rerun()
        try:
            entries = load_entries()
            if not entries:
                st.info("Noch keine Einträge.")
            else:
                entries_sorted = sorted(entries, key=lambda x: -int(x.get('Accuracy',0)))
                medals = ['🥇','🥈','🥉']
                for i, e in enumerate(entries_sorted):
                    a = int(e.get('Accuracy', 0))
                    medal = medals[i] if i < 3 else f"#{i+1}"
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.markdown(f"**{medal} {e.get('Name','—')}**")
                        st.caption(f"w1={e.get('w1')} · w2={e.get('w2')} · bias={e.get('bias')}")
                    with c2:
                        st.metric("", f"{a}/8", label_visibility="collapsed")
                    with c3:
                        if st.button("✕", key=f"del_{i}_{e.get('Name','')}", help="Löschen"):
                            delete_entry(str(e.get('Name',''))); st.rerun()
                    st.progress(int(a/8*100))
                    st.divider()
        except Exception as ex:
            st.error(f"Fehler: {ex}")

# ════════════════════════════════════════════════════════
# TAB 2 — WAS LERNT DIE KI?
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Was haben die Gewichte gelernt?")
    st.markdown("Die KI hat nicht einfach Daten gespeichert — sie hat eine **Regel** gelernt. Die Gewichte verraten uns welche Features wichtig sind.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("##### Deine aktuellen Gewichte")

        opt_w1, opt_w2, opt_b = 0.5, 1.5, -2.2

        # Weight comparison bar chart
        fig_w = go.Figure()
        categories = ['w1 (Verspätung)', 'w2 (Rushhour)', 'Bias']
        your_vals   = [w1, w2, bias]
        opt_vals    = [opt_w1, opt_w2, opt_b]

        fig_w.add_trace(go.Bar(
            name='Deine Gewichte', x=categories, y=your_vals,
            marker_color='#1a3f6f', text=[f"{v:.2f}" for v in your_vals],
            textposition='outside'
        ))
        fig_w.add_trace(go.Bar(
            name='Optimale Gewichte', x=categories, y=opt_vals,
            marker_color='#EB0000', opacity=0.6,
            text=[f"{v:.2f}" for v in opt_vals], textposition='outside'
        ))
        fig_w.update_layout(
            barmode='group', plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=20,r=20,t=40,b=40), height=300,
            yaxis=dict(gridcolor='#f0f0f0', zeroline=True, zerolinecolor='#ccc')
        )
        st.plotly_chart(fig_w, use_container_width=True, config={"displayModeBar":False})

    with col2:
        st.markdown("##### Was die optimalen Gewichte sagen")
        st.info(f"""
**w1 = {opt_w1}** — Jede Minute Verspätung zählt mit Gewicht 0.5

**w2 = {opt_w2}** — Rushhour zählt dreimal mehr als eine Minute Verspätung

**Bias = {opt_b}** — Die Grundschwelle: ohne Verspätung und ohne Rushhour ist der Zug pünktlich

→ **Rushhour ist der stärkste Faktor.** Ein Zug in der Rushhour braucht fast keine Verspätung um zu spät anzukommen.
        """)

        st.markdown("##### Formel mit optimalen Werten")
        st.code("z = 0.5 × Verspätung + 1.5 × Rushhour − 2.2", language=None)
        st.markdown("""
| Beispiel | z | Vorhersage |
|---|---|---|
| 0 Min., kein Rushhour | 0.5×0 + 1.5×0 − 2.2 = **−2.2** | ✅ Pünktlich |
| 7 Min., Rushhour | 0.5×7 + 1.5×1 − 2.2 = **+2.8** | 🔴 Verspätung |
| 3 Min., kein Rushhour | 0.5×3 + 1.5×0 − 2.2 = **−0.7** | ✅ Pünktlich |
        """)

    st.divider()
    st.markdown("##### Vergleich: dein Modell vs. optimal")
    your_acc = calc_accuracy(w1, w2, bias)
    opt_acc  = calc_accuracy(opt_w1, opt_w2, opt_b)
    c1, c2, c3 = st.columns(3)
    c1.metric("Deine Accuracy (Training)", f"{your_acc}/8", f"{int(your_acc/8*100)}%")
    c2.metric("Optimale Accuracy (Training)", f"{opt_acc}/8", f"{int(opt_acc/8*100)}%")
    delta = your_acc - opt_acc
    c3.metric("Differenz", f"{delta:+d} Züge")

# ════════════════════════════════════════════════════════
# TAB 3 — OVERFITTING
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("Underfitting vs. Overfitting")
    st.markdown("""
Wir haben jetzt **50 simulierte Zugfahrten** — mit einem komplexeren Muster:
Züge können trotz mittlerer Verspätung pünktlich sein (Fahrer holt auf), oder trotz kleiner Verspätung zu spät (Störung, Wetter).
Das Muster ist **nicht mehr linear trennbar**.

Beobachte wie sich Trainingsfehler und Testfehler verändern wenn das Modell komplexer wird.
    """)

    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        complexity = st.slider(
            "Modell-Komplexität",
            min_value=1, max_value=10, value=1, step=1,
            help="1 = einfache Gerade | 3-4 = gut | 7+ = Overfitting"
        )
        labels_map = {
            1:  "📏 Grad 1 — Gerade. Zu simpel.",
            2:  "📐 Grad 2 — Leichte Kurve.",
            3:  "✅ Grad 3 — Gute Balance.",
            4:  "✅ Grad 4 — Gute Balance.",
            5:  "⚠️ Grad 5 — Wird komplex.",
            6:  "⚠️ Grad 6 — Zu komplex.",
            7:  "🔴 Grad 7 — Overfitting beginnt.",
            8:  "🔴 Grad 8 — Starkes Overfitting.",
            9:  "🔴 Grad 9 — Auswendig gelernt.",
            10: "🔴 Grad 10 — Modell ist unbrauchbar auf neuen Daten.",
        }
        st.caption(labels_map.get(complexity, ""))

        show_test = st.toggle("Testdaten einblenden (ungesehene Züge)", value=False)

        fig_fit, train_acc, test_acc = make_fitting_plot(complexity, show_test)
        st.plotly_chart(fig_fit, use_container_width=True, config={"displayModeBar":False})

    with col2:
        st.markdown("##### Trainingsfehler vs. Testfehler")

        train_err = round((1 - train_acc) * 100, 1)
        test_err  = round((1 - test_acc)  * 100, 1)

        c1, c2 = st.columns(2)
        c1.metric("Trainingsfehler", f"{train_err}%",
                  delta=f"{train_err-50:.0f}% vs. Start" if train_err < 50 else None,
                  delta_color="inverse")
        c2.metric("Testfehler", f"{test_err}%",
                  delta="⚠️ hoch" if test_err > 35 else "✅ ok",
                  delta_color="inverse" if test_err > 35 else "normal")

        st.progress(int(train_acc*100), text=f"Training Accuracy: {int(train_acc*100)}%")
        st.progress(int(test_acc*100),  text=f"Test Accuracy:     {int(test_acc*100)}%")

        st.divider()

        if complexity <= 2:
            st.error("""
**Underfitting**

Das Modell ist zu simpel. Eine Gerade kann das Muster nicht erfassen.
Sowohl Trainings- als auch Testfehler sind hoch.
            """)
        elif complexity <= 5:
            st.success("""
**Gute Balance**

Das Modell hat das Muster gelernt — nicht auswendig, sondern als Regel.
Trainingsfehler und Testfehler sind beide niedrig.
Das ist das Ziel.
            """)
        else:
            st.error("""
**Overfitting**

Das Modell hat die Trainingsdaten auswendig gelernt.
Trainingsfehler → 0%. Testfehler → hoch.
Auf neuen Zügen versagt es — weil es keine Regel gelernt hat, sondern Punkte memoriert.
            """)

        st.divider()
        # Error curve
        st.markdown("##### Fehler-Kurve über Komplexität")
        complexities = list(range(1, 11))
        train_accs, test_accs = [], []
        for c in complexities:
            _, ta, tea = make_fitting_plot(c, False)
            train_accs.append((1-ta)*100)
            test_accs.append((1-tea)*100)

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=complexities, y=train_accs, mode='lines+markers',
            name='Trainingsfehler', line=dict(color='#1a3f6f', width=2),
            marker=dict(size=8)
        ))
        fig_curve.add_trace(go.Scatter(
            x=complexities, y=test_accs, mode='lines+markers',
            name='Testfehler', line=dict(color='#EB0000', width=2),
            marker=dict(size=8)
        ))
        fig_curve.add_vline(x=complexity, line_dash="dash", line_color="#f59e0b", line_width=2)
        fig_curve.update_layout(
            xaxis_title='Modell-Komplexität',
            yaxis_title='Fehler (%)',
            plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=30,r=10,t=30,b=40), height=240,
            xaxis=dict(tickvals=complexities, gridcolor='#f0f0f0'),
            yaxis=dict(gridcolor='#f0f0f0', range=[0,60]),
            font=dict(size=11)
        )
        st.plotly_chart(fig_curve, use_container_width=True, config={"displayModeBar":False})
