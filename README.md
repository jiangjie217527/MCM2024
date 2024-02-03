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