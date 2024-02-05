import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

st.set_page_config(
    layout="wide"
)
st.markdown(
    f"""
    <style>
    .stApp{{
        background-image:url(https://pic.imgdb.cn/item/65c0f7db9f345e8d03268dfd.png);
        background-size:cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <p>Team 2424398</p>
    <b style="font-size:70px">Visualization of Model About Ecosystem</b>
    <p style="font-size:15px">Idea about 2024 MCM Problem A</p>
    <hr style="border: 1px solid black">
    """,
    unsafe_allow_html=True
)

st.title("Introduction about this page")

st.markdown(
    f"""
    <p style="font-size:20px">After task1, task2, task3, we obtained some formuals and models for ecosystem. Using these equations, we can easily use different parameters and quickly calcute the changes of number of these species and sex ratio:</p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    $$
    \\frac{{d{N_{prey}}}}{{dt}} = {r_{prey}}{N_{prey}}(1 - \\frac{{{N_{prey}}}}{{K(R)}}) - \\alpha {N_{prey}}{N_{pred}}\\\\
    $$

    $$
    \\frac{{dF}}{{dt}} =  - 2 * (1-\eta) * {r_{female}}F{\\rm{ + }}{\\beta _{female}}F*{N_{prey}} + {k_{male}}M - human_{cache} * F\\\\
    $$

    $$
    \\frac{{dM}}{{dt}} =  - 2 * \eta * {r_{male}}{\\rm{M + }}{\\beta _{male}}M*{N_{prey}} + {k_{female}}F - human_{cache} * M\\\\
    $$
    """
)

st.markdown(
    f"""
    <p style="font-size:20px">With these formulas and inner parameters, together with the parameters on your left hand, we can get these diagrams</p>
    <p style="font-size:20px">You can drag the slidebar to change these parameters.</p>
    <hr style="border: 1px solid black">
    """,
    unsafe_allow_html=True
)

st.sidebar.write("Change the parameters here, wait for one second and observe the difference in these curves")
# st.sidebar.write('time3s')

# Get user input for frequency and amplitude
with st.sidebar:
    st.markdown("$\eta$")

eta = st.sidebar.slider('this parameter controls sex ratio. When this is set to 0.5, the sex ratio is fixed at 0.7',
                        min_value=0.4, max_value=0.6, value=0.4)
with st.sidebar:
    st.markdown("**N_prey**")

N_prey0 = st.sidebar.slider('init number of prey divided by 1e5 at time 0', min_value=4.0, max_value=10.0, value=8.0)
with st.sidebar:
    st.markdown("**N_pred**")

N_pred0 = st.sidebar.slider('init number of lampery divided by 1e4 at time 0', min_value=0.6, max_value=1.6, value=1.0)
with st.sidebar:
    st.markdown("**human cache**")

human = st.sidebar.slider(
    '0 means people hardly cache lampery as food, while 1 means people cathe nearly all the lampery as food!',
    min_value=0.0, max_value=1.0, value=0.0)
K_R = 1508488
K_F = 468700000000000000000  # 环境承载力
r_prey = 0.964567701518982
alpha = 6.82544520966953e-6
r = 2.0213353573062
beta = 1.50939093249243e-6
k_M = 0.538704624743195
k_F = 7 / 3 * 7 / 3 * k_M
t_min = 0  # start at year 0
t_max = 25  # end at year 10 (1971)
t_h = 1e-1


# 第0年为1962年

def funcNt(n, f, m):
    return r_prey * n * (1 - n / K_R) - alpha * n * (f + m)


def funcFt(n, f, m, eta):
    return -2 * (1 - eta) * r * f + beta * f * n + k_M * m - human * f * 2


def funcMt(n, m, f, eta):
    return -2 * eta * r * m + beta * m * n + k_F * f - human * m * 2


def rat(m, f):
    return m / (m + f)


def work(eta):
    t = np.linspace(t_min, t_max, int((t_max - t_min) / t_h + 1))
    n = t.copy()
    f = t.copy()
    m = t.copy()
    ratio = t.copy()
    n[0] = N_prey0 * 1e5
    f[0] = N_pred0 * 0.3 * 1e4
    m[0] = N_pred0 * 0.7 * 1e4
    ratio[0] = rat(m[0], f[0])
    # h = t_h
    for i in range(t.shape[0] - 1):
        k1_n = funcNt(n[i], f[i], m[i])
        k1_f = funcFt(n[i], f[i], m[i], eta)
        k1_m = funcMt(n[i], m[i], f[i], eta)

        k2_n = funcNt(n[i] + k1_n * t_h / 2.0, f[i], m[i])
        k2_f = funcFt(n[i], f[i] + k1_f * t_h / 2.0, m[i], eta)
        k2_m = funcMt(n[i], m[i] + k1_m * t_h / 2.0, f[i], eta)

        k3_n = funcNt(n[i] + k2_n * t_h / 2.0, f[i], m[i])
        k3_f = funcFt(n[i], f[i] + k2_f * t_h / 2.0, m[i], eta)
        k3_m = funcMt(n[i], m[i] + k2_m * t_h / 2.0, f[i], eta)

        k4_n = funcNt(n[i] + k3_n * t_h, f[i], m[i])
        k4_f = funcFt(n[i], f[i] + k3_f * t_h, m[i], eta)
        k4_m = funcMt(n[i], m[i] + k3_m * t_h, f[i], eta)

        n[i + 1] = n[i] + t_h / 6.0 * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n)
        f[i + 1] = f[i] + t_h / 6.0 * (k1_f + 2.0 * k2_f + 2.0 * k3_f + k4_f)
        m[i + 1] = m[i] + t_h / 6.0 * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m)
        ratio[i + 1] = rat(m[i + 1], f[i + 1])
        # print(k1_f,k2_f,k3_f,k4_f)
    data = {
        't': t,
        'N_prey': n,
        'F': f,
        'M': m,
        'ratio(M/(F+M))': ratio
    }
    df = pd.DataFrame(data)
    # st.line_chart(df,x="t",y="N_prey",color='#D6AFB9',height =280)
    # st.line_chart(df,x="t",y=["F","M"],color=['#D6AFB9','#7E9BB7'],height =280)
    # st.line_chart(df,x="t",y="ratio(M/(F+M))",color='#D6AFB9',height =280)
    # plt.figure(figsize=(6,6))
    # plt.figure(figsize=(6,12))
    st.title("output")
    st.markdown(
        f"""
    <p style="font-size:25px">first(left) diagram represents the change of the number of prey.</p>
    """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
    <p style="font-size:25px">second(right) diagram represents the change of the number of male and female lampery.</p>
    """,
        unsafe_allow_html=True
    )
    c1, c2 = st.columns(spec=2)
    fig, ax = plt.subplots()
    plt.subplot(1, 1, 1)
    plt.plot(t, n, '#D6AFB9', label='N_prey', linewidth=1.5)
    plt.legend()  # prop = {'size':18}
    plt.xlabel('t_year')  # ,fontsize=18
    plt.ylabel('number')
    plt.xticks()
    plt.yticks()
    plt.title('N_prey-t')
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    c1.image(buf)

    fig, ax = plt.subplots()
    plt.subplot(1, 1, 1)
    plt.plot(t, f, '#D6AFB9',
             label='F', linewidth=1.5)
    plt.plot(t, m, '#7E9BB7',
             label='M', linewidth=1.5)
    plt.legend()
    plt.xlabel('t_year')
    plt.ylabel('number')
    plt.xticks()
    plt.yticks()
    plt.title('n-t')
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    c2.image(buf)
    # st.pyplot(fig)

    st.markdown(
        f"""
    <p style="font-size:25px">third diagram represents the change of the sex ratio.</p>
    """,
        unsafe_allow_html=True
    )
    fig, ax = plt.subplots()
    plt.subplot(1, 1, 1)
    plt.plot(t, ratio, '#D6AFB9',
             label='ratio', linewidth=1.5)
    plt.legend()
    plt.xlabel('t_year')
    plt.ylabel('ratio(M/(M+F))')
    plt.xticks()
    plt.yticks()
    plt.title('ratio-t')
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    c1, c2 = st.columns(spec=2)
    c1.image(buf)
    # st.pyplot(fig)


work(eta)

