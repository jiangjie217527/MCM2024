# Problem A: Resource Availability and Sex Ratios

Object: sea lampreys

location: lake or sea habitats and migrate up rivers to spawn

external circumstances: how quickly they grow during the larval stage -> availability of food

e.g. low food availability -> lower growth rate -> 78% male


task: examine the advantages and disadvantages of the ability for a species to alter its sex ratio depending on resource availability

question:

1. impact on the larger ecological system
2. advantages and disadvantages to the population of lampreys
3. impact on the stability of the ecosystem
4. ecosystem offer advantages to others

to define:

1. impact
2. larger ecological system
3. advantages(disadvantages) mean raising(reducing) the population of lampreys, which depends on rate of birth and death(not consider move in and out)
4. stability of the ecosystem(the abality of defending and reviving from disaster, biodiversity and so on)

to solve:

1. advantages means more population? more food and healther?(remember wolf and sheep)
2. calculate matters
3. lacking data

模型

1. 竞争模型
2. 逻辑斯蒂模型
3. 捕食者模型(寄生虫)
4. 捕捞模型(北美)

公式

$$
\frac{{d{N_{prey}}}}{{dt}} = {r_{prey}}{N_{prey}}(1 - \frac{{{N_{prey}}}}{{K(R)}}) - \alpha {N_{prey}}{N_{pred}}\\
\frac{{dF}}{{dt}} = -2*(1-\eta){r_{female}}F(1 - \frac{F}{{{K_F}}}){\rm{ + }}{\beta _{female}}F*{N_{prey}} + {k_{male}}M\\
\frac{{dM}}{{dt}} = -2*\eta * {r_{male}}{\rm{M(1 - }}\frac{M}{{{K_M}}}{\rm{) + }}{\beta _{male}}M*{N_{prey}} + {k_{female}}F\\
N_{prey}' = f(N_{prey},F,M) = l(N_{prey},F)\\
F_{prey}' = g(F,N_{prey},M) = h(N_{prey},F)\\
(M = 0.7*N_{pred},M = 0.3*N_{pred})\\
\frac{{dF}}{{dt}} = {r_{female}}F(1 - \frac{F}{{{K_F}}}){\rm{ + }}{\beta _{female}}F*{N_{prey}} + {k_{male}}M\\
$$

number of trout 
$$
392100\\
775161\\
1185755\\

22825 -> 45650\\
35980 -> 71960\\ 
5610 -> 11220\\
7267 -> 14534\\
4922 -> 9844\\
71960\\ 
11220\\
14534\\
9844
$$

数据假设：


1. male percentage = 70%


0.534225266,0.021719938,0.266805593,6
0.496012859,0.132056549,1.066573509,11.3
0.3588253,0.048297692,1.587176431,9.68
0.237218888,0.62427921,0.105365189,8.5

6,	1/0.534225266,	0.021719938,	0.266805593

1/0.496012859,	11.3,	1.066573509,	0.132056549

0.048297692,	1.587176431,	9.68,	0.3588253

0.105365189,	0.62427921,	0.237218888,	8.5

~~~
# Interactive Streamlit elements, like these sliders, return their value.
# This gives you an extremely simple interaction model.
iterations = st.sidebar.slider("Level of detail", 2, 20, 10, 1)
separation = st.sidebar.slider("Separation", 0.7, 2.0, 0.7885)

# Non-interactive elements return a placeholder to their location
# in the app. Here we're storing progress_bar to update it later.
progress_bar = st.sidebar.progress(0)

# These two elements will be filled in later, so we create a placeholder
# for them using st.empty()
frame_text = st.sidebar.empty()
image = st.empty()

m, n, s = 960, 640, 400
x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))

for frame_num, a in enumerate(np.linspace(0.0, 4 * np.pi, 100)):
    # Here were setting value for these two elements.
    progress_bar.progress(frame_num)
    frame_text.text("Frame %i/100" % (frame_num + 1))

    # Performing some fractal wizardry.
    c = separation * np.exp(1j * a)
    Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
    C = np.full((n, m), c)
    M: Any = np.full((n, m), True, dtype=bool)
    N = np.zeros((n, m))

    for i in range(iterations):
        Z[M] = Z[M] * Z[M] + C[M]
        M[np.abs(Z) > 2] = False
        N[M] = i

    # Update the image placeholder by calling the image() function on it.
    image.image(1.0 - (N / N.max()), use_column_width=True)

# We clear elements by calling empty on them.
progress_bar.empty()
frame_text.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
~~~

~~~
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
~~~

# Problem B: Searching for Submersibles

Object submersibles & safety procedures

location: bottom of the Ionian Sea

external circumstances: a loss of communication to the host ship and possible mechanical defects

task: develop a model to predict the location of the submersible over time

e.g. sea floor or at some point of neutral buoyancy underwater, be affected by currents, differing densities in the sea  ,geography of the sea floor

question:

1. Loacte: uncertainties, information to decrease these uncertainties prior to an incident? equipment.
2. Prepare: search equipment on the host ship(different types, costs with availability, maintenance, readiness, usage) additional equipment on the rescue vessel
3. Search: initial points of deployment and search patterns. probability of finding the submersible as a function of time.
4. Extrapolate: other tourist destinations,change to account for multiple submersibles