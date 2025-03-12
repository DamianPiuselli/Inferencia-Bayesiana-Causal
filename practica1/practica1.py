import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import inspect

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
    else:
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

    res = num / den  # P(s|c,M) = P(s,c|M)/P(c|M)

    return res


def pr_csM(r, c, s, m):
    # Predicción del segundo dato dado el primero
    num = prcs_M(r, c, s, m)  # p(r,c,s|M)

    den = 0  # p(c,s|M) = sum_r P(r,c,s|M)
    for _r in H:
        den += prcs_M(_r, c, s, m)

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
    return [pEpisodio_M(c, s, r, m) for c, s, r in Datos]


def pDatos_M(Datos, m):
    # P(Datos = {(c0,s0,r0),(c1,s1,r1),...} | M )
    return np.prod(_secuencia_de_predicciones(Datos, m))


pDatos_M(Datos, m=0)  # 8.234550899283273e-21
pDatos_M(Datos, m=1)  # 3.372872048346429e-17

print(pDatos_M(Datos, m=0))
print(pDatos_M(Datos, m=1))

# 1.4 Calcular predicción de los datos P(Datos)


def pM(m):
    # Prior de los modelos

    # asumo que ambos modelos son igual de probables a priori
    return 0.5


def pDatos(Datos):
    # sum_m P(Datos,M=m)
    # sum_m P(Datos|M=m)P(M=m)

    return sum([pDatos_M(Datos, m) * pM(m) for m in [0, 1]])


# 1.5 Posterior de los modelos


def pM_Datos(m, Datos):
    # P(M|Datos = {(c0,s0,r0),(c1,s1,r1),...})
    return pDatos_M(Datos, m) * pM(m) / pDatos(Datos)


# 1.6 Graficar


def lista_pM_Datos(m, Datos):
    # [P(M | (c0,s0,r0) ), P(M | (c0,s0,r0),(c1,s1,r1) ), ... ]

    return [pM_Datos(m, Datos[:t]) for t in range(0, len(Datos))]


plt.plot(lista_pM_Datos(m=0, Datos=Datos), label="M0: Base")
plt.plot(lista_pM_Datos(m=1, Datos=Datos), label="M1: Monty Hall")
plt.legend()
plt.show()


# # 2.1


# def pp_Datos(p, Datos):
#     # P(p | Datos = {(c0,s0,r0),(c1,s1,r1), ... })
#     return NotImplementedError(
#         f"La función {inspect.currentframe().f_code.co_name}() no está implementada"
#     )


# # 2.2


# def pEpisodio_DatosMa(Episodio, Datos):
#     # P(EpisodioT = (cT, sT, rT) | Datos = {(c0,s0,r0),(c1,s1,r1), ... })
#     cT, sT, rT = Episodio
#     return NotImplementedError(
#         f"La función {inspect.currentframe().f_code.co_name}() no está implementada"
#     )


# # 2.3

# # Actualizar pDatos_M(Datos, m, log = False) agregándole un parámetro log

# # 2.4


# def log_Bayes_factor(log_pDatos_Mi, log_pDatos_Mj):
#     # Recibe la predicción de los datos en ordenes de magnitud
#     # y devuelve el logaritmo del Bayes factor, es decir,
#     # la diferencia de predicciones en órdenes de magnitud
#     return NotImplementedError(
#         f"La función {inspect.currentframe().f_code.co_name}() no está implementada"
#     )


# # 2.5


# def geometric_mean(Datos, m, log=False):
#     # Dado los datos y el modelo devuelve la media geométrica
#     return NotImplementedError(
#         f"La función {inspect.currentframe().f_code.co_name}() no está implementada"
#     )


# # 2.6

# # actualizar pM_Datos(m,Datos) para que soporte al modelo alternativo
