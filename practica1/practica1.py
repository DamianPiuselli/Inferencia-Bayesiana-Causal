import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import inspect
import os

H = np.arange(3)  # Posibles valores de las hipótesis
# Como estamos trabajando en python vamos empezar con 0
# Es decir, la posición del regalo r \in {0,1,2} y de
# la misma forma con el resto de las variables.


def pr(r):  # P(r)
    return 1 / 3


def pc(c):  # P(c)
    return 1 / 3


def ps_rM0(s, r):  # P(s|r,M=0)
    # la pista no puede ser el regalo
    if s == r:
        return 0
    # la pista es cualquiera de las cajas restantes con probabilidad 1/2
    return 1 / 2


def ps_rcM1(s, r, c):  # P(s|r,c,M=1)
    # la pista no puede ser el reservado
    if s == c:
        return 0

    # la pista no puede ser el regalo
    if s == r:
        return 0

    # quedan dos tipos de escenario c==r y c!=r
    # si  c==r, la pista puede ser cualquiera de las cajas restantes con probabilidad 1/2
    if c == r:
        return 1 / 2

    # si c!=r, la pista es la caja restante que no contiene el regalo con probabilidad 1

    if c != r:
        return 1


def prcs_M(r, c, s, m):  # P(r,c,s|M)
    # producto de las condicionales

    if m == 0:
        return pr(r) * pc(c) * ps_rM0(s, r)
    if m == 1:
        return pr(r) * pc(c) * ps_rcM1(s, r, c)


def ps_cM(s, c, m):
    # Predicción del segundo dato dado el primero
    num = 0  # P(s,c|M) = sum_r P(r,c,s|M)
    for _r in H:
        num += prcs_M(_r, c, s, m)

    den = 0  # P(c|M) = sum_{rs} P(r,c,s|M)
    for _r in H:
        for _s in H:
            den += prcs_M(_r, c, _s, m)

    if num == 0:
        return 0

    res = num / den  # P(s|c,M) = P(s,c|M)/P(c|M)

    return res


def pr_csM(r, c, s, m):
    # Predicción del segundo dato dado el primero
    num = prcs_M(r, c, s, m)  # p(r,c,s|M)

    den = 0  # p(c,s|M) = sum_r P(r,c,s|M)
    for _r in H:
        den += prcs_M(_r, c, s, m)

    if num == 0:
        return 0

    res = num / den  # P(r|c,s,M) = P(r,c,s|M)/p(c,s|M)
    return res


def pEpisodio_M(c, s, r, m):  # P(Datos = (c,s,r) | M)
    # Predicción del conjunto de datos P(c,s,r|M)

    # P(c,s,r|M) =  P(r|c,s,M)P(s|c,M)*P(c|M)
    # con P(c|M) = P(c) = 1/3  (eleccion es independiente del modelo)

    return pr_csM(r, c, s, m) * ps_cM(s, c, m) * pc(c)


# 1.2 Simular datos con Monty Hall


def simular(T=16, seed=0):
    np.random.seed(seed)
    Datos = []
    for t in range(T):
        # regalo en posicion al azar
        r = np.random.choice(3, p=[pr(hr) for hr in H])
        # primera eleccion de la caja al azar
        c = np.random.choice(3, p=[pc(cr) for cr in H])
        # elegir entre las opciones restantes h!=r y h!=c con probabilidad ps_rcM1
        s = np.random.choice(3, p=[ps_rcM1(h, r, c) for h in H])

        Datos.append((c, s, r))
    return Datos


T = 16
Datos = simular()


# 1.3 Predicción P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M )


def _secuencia_de_predicciones(Datos, m):
    # Si se guarda la lista de predicciones de cada uno
    # de los episodios [P(Episodio0|M),P(Episodio1|M),... ]
    # esto va a servir tanto para calcular la predicción
    # P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M ),
    # pero también va a servir después para graficar como
    # va cambiando el posterior de los modelos en el tiempo
    if m in [0, 1]:
        return [pEpisodio_M(c, s, r, m) for c, s, r in Datos]
    if m == "a":
        return [pEpisodio_DatosMa((c, s, r), Datos) for c, s, r in Datos]
    raise ValueError("Modelo no reconocido")


def pDatos_M(Datos, m, log=False):
    # P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M )
    if log:
        return np.sum(np.log10(_secuencia_de_predicciones(Datos, m)))
    return np.prod(_secuencia_de_predicciones(Datos, m))


pDatos_M(Datos, m=0)  # 8.234550899283273e-21
pDatos_M(Datos, m=1)  # 3.372872048346429e-17

print(pDatos_M(Datos, m=0))
print(pDatos_M(Datos, m=1))

# 1.4 Calcular predicción de los datos P(Datos)


def pM(m, considerar_alternativo=False):
    # Prior de los modelos
    # asumo que ambos modelos son igual de probables a priori

    return 1 / 3 if considerar_alternativo else 0.5


def pDatos(Datos, considerar_alternativo=False):
    # sum_m P(Datos,M=m)
    # sum_m P(Datos|M=m)P(M=m)

    modelos = [0, 1, "a"] if considerar_alternativo else [0, 1]

    return sum([pDatos_M(Datos, m) * pM(m, considerar_alternativo) for m in modelos])


# 1.5 Posterior de los modelos


def pM_Datos(m, Datos, considerar_alternativo=False):
    # P(M|Datos = {(c0,s0,r0),(c1,s1,r1),...})

    return (
        pDatos_M(Datos, m)
        * pM(m, considerar_alternativo)
        / pDatos(Datos, considerar_alternativo)
    )


# 1.6 Graficar


def lista_pM_Datos(m, Datos, considerar_alternativo=False):
    # [P(M | (c0,s0,r0) ), P(M | (c0,s0,r0),(c1,s1,r1) ), ... ]

    return [
        pM_Datos(m, Datos[:t], considerar_alternativo) for t in range(0, len(Datos))
    ]


plt.plot(lista_pM_Datos(m=0, Datos=Datos), label="M0: Base")
plt.plot(lista_pM_Datos(m=1, Datos=Datos), label="M1: Monty Hall")
plt.legend()
plt.show()


# # 2.1


## las proximas funciones las agrego yo por conveniencia.
def pa_p(a_t, p):
    # P(a_t|p), a_t boolean
    return p if a_t else 1 - p


def pcst_p(c_t, s_t, r_t, p):
    # P(c,s,r|p)
    output = 0

    # integracion sobre a_t
    for a_t in [0, 1]:
        # a_t == 0, no recuerda la caja reservada, modelo M0
        if a_t == 0:
            output += pr(r_t) * pc(c_t) * pa_p(a_t, p) * ps_rM0(s_t, r_t)
        # a_t == 1, recuerda la caja reservada, modelo M0
        if a_t == 1:
            output += pr(r_t) * pc(c_t) * pa_p(a_t, p) * ps_rcM1(s_t, r_t, c_t)
    return output


# P(p), prior de la probabilidad p, asumo uniforme en el intervalo 0,1
def pp(p):
    if p < 0 or p > 1:
        return 0
    return 1


def pp_Datos(p, Datos):
    # P(p | Datos = {(c0,s0,r0),(c1,s1,r1), ... })
    # p :: probabilidad de recordar la caja reservada

    num = np.prod([pcst_p(c_t, s_t, r_t, p) for c_t, s_t, r_t in Datos]) * pp(p)
    den = sum(
        np.prod([pcst_p(c_t, s_t, r_t, _p) for c_t, s_t, r_t in Datos]) * pp(_p)
        for _p in np.linspace(0, 1, 11)
    )

    if num == 0:
        return 0

    return num / den


# cargar NoMontyHall.csv
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "NoMontyHall.csv")
datos_NoMonty = pd.read_csv(csv_path)[:60].to_numpy()  # primeros 60 datos

# Grafico para ver que la pp se comporta como espero
plt.plot(
    np.linspace(0, 1, 50),
    [pp_Datos(_p, datos_NoMonty) for _p in np.linspace(0, 1, 50)],
)
plt.title(
    "Probabilidad posterior de p para los primeros 60 datos.\n Se asume un prior uniforme sobre p en el (0,1)"
)
plt.xlabel("p")
plt.ylabel("P(p | Datos)")
plt.show()


# 2.2


def pEpisodio_DatosMa(Episodio, Datos):
    # P(EpisodioT = (cT, sT, rT) | Datos = {(c0,s0,r0),(c1,s1,r1), ... })
    cT, sT, rT = Episodio

    # Integrar sobre p,   P(cT, sT, rT | p) * P(p | Datos)
    posterior_prob = 0
    for p in np.linspace(0, 1, 11):
        posterior_prob += pcst_p(cT, sT, rT, p) * pp_Datos(p, Datos)
    return posterior_prob


# # 2.3

# # Actualizar pDatos_M(Datos, m, log = False) agregándole un parámetro log


# # 2.4


def log_Bayes_factor(log_pDatos_Mi, log_pDatos_Mj):
    # Recibe la predicción de los datos en ordenes de magnitud
    # y devuelve el logaritmo del Bayes factor, es decir,
    # la diferencia de predicciones en órdenes de magnitud
    return log_pDatos_Mi - log_pDatos_Mj


log_pDatos_M0 = pDatos_M(datos_NoMonty, m=0, log=True)
print(f"Logaritmo de P(Datos|M0): {log_pDatos_M0:.5e}")

# el modelo de Monty Hall da una loglikelihood de -inf debido a que la verosimiltud se hace cero cuando ocurre el primer olvido
# y se produce un dato que no es compatible con el modelo causal.

log_pDatos_M1 = pDatos_M(datos_NoMonty, m=1, log=True)
print(f"Logaritmo de P(Datos|M1): {log_pDatos_M1:.5e}")

log_pDatos_MA = pDatos_M(datos_NoMonty, m="a", log=True)
print(f"Logaritmo de P(Datos|MA): {log_pDatos_MA:.5e}")


# # 2.5


def geometric_mean(Datos, m, log=False):
    # Dado los datos y el modelo devuelve la media geométrica
    if log:
        return np.mean(np.log10(_secuencia_de_predicciones(Datos, m)))
    return np.prod(_secuencia_de_predicciones(Datos, m)) ** (1 / len(Datos))


print(
    "Logaritmo de la media geométrica de las predicciones de los datos para M0: ",
    geometric_mean(datos_NoMonty, m=0, log=True),
)
print(
    "Logaritmo de la media geométrica de las predicciones de los datos para M1: ",
    geometric_mean(datos_NoMonty, m=1, log=True),
)

print(
    "Logaritmo de la media geométrica de las predicciones de los datos para MA: ",
    geometric_mean(datos_NoMonty, m="a", log=True),
)
# # 2.6

# # actualizar pM_Datos(m,Datos) para que soporte al modelo alternativo

plt.plot(
    lista_pM_Datos(m=0, Datos=datos_NoMonty, considerar_alternativo=True),
    label="M0: Base",
)
plt.plot(
    lista_pM_Datos(m=1, Datos=datos_NoMonty, considerar_alternativo=True),
    label="M1: Monty Hall",
)
plt.plot(
    lista_pM_Datos(m="a", Datos=datos_NoMonty, considerar_alternativo=True),
    label="MA: Modelo Alternativo",
)
plt.legend()
plt.show()
