import math
import random
import inspect
import warnings

justificacion = {}

# El examen consiste en distribuir creencias entre las hipótesis mutuamente contradictorias en cada una de las afirmaciones o pregunta que se realizan.


def es_distribucion_de_creencias(respuestas, funcion, enunciado):
    """
    Verifica que la respuesta sea una distribución de creencias.
    """
    suma_1 = math.isclose(sum([respuestas[r] for r in respuestas]), 1.0)
    positivas_o_nulas = math.isclose(sum([respuestas[r] < 0 for r in respuestas]), 0.0)
    if not (suma_1 and positivas_o_nulas):
        warnings.warn(
            f"""La respuesta al enunciado {funcion} '{enunciado}' no es una distribución de creencias""",
            RuntimeWarning,
        )

    return suma_1 and positivas_o_nulas


def maxima_incertidumbre(respuestas):
    """
    Si no se proveé una respuesta que sea una distribución de creencias, se construye una dividiendo la creencia en partes iguales.
    """
    n = len(respuestas)
    return {r: 1 / n for r in respuestas}


#######################################
##### Selección Múltiple Semana 3 #####

random.seed(0)


def _3_1(
    enunciado="""Una casa de apuestas paga 3 por Cara y 1.2 por Sello. La moneda tiene 0.5 de probabilidad de que salga Cara o Sello. Si no estamos obligados a apostar todos nuestros recursos cada vez que jugamos, ¿qué proporción conviene apostar a Cara, qué proporción a Sello y qué proporción ahorramos?.""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}

    # Proporción de los recursos que apostamos a Cara
    respuestas["Cara"] = 0.25

    # Proporción de los recursos que apostamos a Sello
    respuestas["Sello"] = 0.0

    # Proporción de los recursos que apostamos a Sello
    respuestas["Ahorro"] = 0.75

    justificacion[
        nombre
    ] = """
    Nunca apostamos a sello ya que paga menos que apostar a cara y ambos son equiprobables. No apostamos todo el capital en una sola apuesta para no exponernos a la ruina.
    La proporcion optima se puede calcular de maximizar la media geometrica.

    """

    # Revisa si es una distribución de creencias (que sume 1)
    valida = es_distribucion_de_creencias(
        respuestas, inspect.currentframe().f_code.co_name, enunciado
    )

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_2(
    enunciado="""Cuál es la menor cantidad de preguntas Sí/No que se necesitan para identificar un entero de 0 a 15? """,
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["2"] = 0.0
    respuestas["3"] = 0.0
    respuestas["4"] = 1.0
    respuestas["5"] = 0.0
    respuestas["6"] = 0.0
    respuestas["7"] = 0.0
    respuestas["8"] = 0.0
    respuestas["9"] = 0.0
    respuestas["10"] = 0.0
    respuestas["11"] = 0.0
    respuestas["12"] = 0.0
    respuestas["13"] = 0.0
    respuestas["14"] = 0.0
    respuestas["15"] = 0.0

    justificacion[
        nombre
    ] = """
    Cada pregunta puede divir el espacio de busqueda en 2, por lo que 4 pregutnas son suficientes para determinar el numero.
    """

    valida = es_distribucion_de_creencias(
        respuestas, inspect.currentframe().f_code.co_name, enunciado
    )

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_3(
    enunciado="""Los modelos se evalúan en función de su capacidad predictiva. El mejor modelo es el que predice los datos con probabilidad 1. ¿Qué tipo de preguntas (o recolección de datos) ofrecen mayor información? ¿Sobre las que no nos generan sorpresa, sobre las que sí nos generan sorpresa o la información de una respuestas (o dato) no depende de la sorpresa?""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Sí nos generan sorpresa"] = 1.0
    respuestas["No nos generan sorpresa"] = 0.0
    respuestas["La sorpresa es independiente de la información"] = 0.0

    justificacion[
        nombre
    ] = """
    El hecho de que genere sorpresa implica que los datos son mas informativos en el contexto del modelo y por lo tanto se puede aprender mas de ellos y aumentar las capacidades del modelo. El caso contrario seria medir un dato para el cual el modelo tiene alta certidumbre y por lo tanto no aporta mucha informacion adicional.
    """

    valida = es_distribucion_de_creencias(
        respuestas, inspect.currentframe().f_code.co_name, enunciado
    )

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_4(
    enunciado="""El juego submarino es una simplificación del juego 'Batalla Naval'. Hay un tablero de 8x8 y solo una de las celdas contiene al submarino. ¿Obtenemos la misma información si encontramos al submarino en el primer intento que en el n-ésimo intento?
 """,
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Sí, obtenemos la misma información"] = 1.0
    respuestas["No, obtenemos mayor o menor información"] = 0.0

    justificacion[
        nombre
    ] = """
    La informacion a obtener en este contexto es la posicion del unico submarino. Una vez que la descubrimos es parte del contexto que el resto de los casilleros son agua y no hay mas informacion por obtener, indepedientemente de cuantas veces hayamos intentado encontrarlo.
    """

    valida = es_distribucion_de_creencias(
        respuestas, inspect.currentframe().f_code.co_name, enunciado
    )

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_5(
    enunciado="""Se dice que un modelo es lineal cuando el modelo solo puede modelar relaciones no lineales entre los datos.""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Verdadero"] = 0.0
    respuestas["Falso"] = 1.0

    justificacion[
        nombre
    ] = """
    Mas alla de que un modelo puede ser lineal en los parametros o en las variables, el enunciado es falso de cualquier manera.
    """

    valida = es_distribucion_de_creencias(
        respuestas, inspect.currentframe().f_code.co_name, enunciado
    )

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_6(
    enunciado="""Al comparar modelos causales alternativos dado un conjunto de datos, P(M|D), dónde está contenida la información de los datos.""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Solo en P(D|M)"] = 0.0
    respuestas["Solo en P(D)"] = 0.0
    respuestas["En ambas en P(D) y P(D|M)"] = 1.0
    respuestas["En todo los elementos de P(M|D)"] = 0.0

    justificacion[
        nombre
    ] = """
    La informacion respectiva a los datos esta contenida en la verosimilitud y en la probabilidad marginal de los datos.
    """

    valida = es_distribucion_de_creencias(respuestas, nombre, enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_7(
    enunciado="""Si la realidad causal subyacente contiene aleatoriedad, ¿contar con el modelo causal probabilístico que se corresponde perfectamente con la realidad causal subyacente permite eliminar la sorpresa completamente (predecir con 1 todos los datos observados)?
 """,
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Sí"] = 0.0
    respuestas["No"] = 1.0

    justificacion[
        nombre
    ] = """
    Si los datos contienen aleatoriedad, nunca se puede predecir con certeza absoluta los datos observados.
    """

    valida = es_distribucion_de_creencias(respuestas, nombre, enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_8(
    enunciado="""¿Podemos tener esperanza de que alguna vez el avance de la Inteligencia Artificial permita desarrollar modelos que mejoren el desempeño de los modelos causales probabilístico que se corresponde perfectamente con la realidad causal subyacente?
 """,
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["Sí"] = 0.1
    respuestas["No"] = 0.9

    justificacion[
        nombre
    ] = """
    Como en la pregunta anterior, si la realidad subyacente contiene aleatoriedad inherente, no se puede predecir con certeza absoluta los datos observados. 
    """

    valida = es_distribucion_de_creencias(respuestas, nombre, enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_9(
    enunciado="""Tenemos 12 pelotas visualmente iguales. Todas tienen el mismo peso, salvo una que tiene un peso distinto al resto, imperceptible para el ser humano, pero que es suficiente para inclinar una balanza mecánica de dos bandejas. Decidir cómo distribuir las 12 pelotas en el primer uso de la balanza (bandeja izquierda, bandeja derecha, afuera) para garantizar que la balanza sea usada la menor cantidad de veces posibles.""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["(6, 6, 0)"] = 0.0
    respuestas["(5, 5, 2)"] = 0.0
    respuestas["(4, 4, 4)"] = 1.0
    respuestas["(3, 3, 6)"] = 0.0
    respuestas["(2, 2, 8)"] = 0.0
    respuestas["(1, 1, 10)"] = 0.0

    justificacion[
        nombre
    ] = """
    con (4,4,4) el peor caso es tener que usar la balanza 3 veces.  
    """

    valida = es_distribucion_de_creencias(respuestas, nombre, enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


def _3_10(
    enunciado="""Supongamos que sabemos de las 12 pelotas solo la 1, 2, 3 o 4 puede ser la pelota que pesa menos, o que la 5, 6, 7 o 8 la pelota que pesa más. Decidir qué pelotas poner en la balanza izquierda y en derecha para garantizar que la balanza sea usada la menor cantidad de veces posibles.""",
):
    nombre = inspect.currentframe().f_code.co_name

    respuestas = {}
    respuestas["{1,2,6} vs {3,4,5}"] = 0.0
    respuestas["{1,2,5,6} vs {3,4,7,8}"] = 0.0
    respuestas["{1,6} vs {3,7}"] = 1.0

    justificacion[
        nombre
    ] = """
"{1,6} vs {3,7}" permite descartar mas pelotas en la primer pesada. Ya sea que se equilibre o no, nos quedamos con 4 pelotas en la segunda pesada."""

    valida = es_distribucion_de_creencias(respuestas, nombre, enunciado)

    if valida:
        return respuestas
    else:
        return maxima_incertidumbre(respuestas)


if __name__ == "__main__":
    print(_3_1())
    print(_3_2())
    print(_3_3())
    print(_3_4())
    print(_3_5())
    print(_3_6())
    print(_3_7())
    print(_3_8())
    print(_3_9())
    print(_3_10())
