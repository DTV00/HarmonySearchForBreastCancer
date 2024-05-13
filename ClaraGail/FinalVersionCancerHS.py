import csv
import numpy as np
import math
import pandas as pd
import random
from scipy.stats import pearsonr

def PreProcess(file):
    df = pd.read_csv('BCSC_risk_factors_153821.csv')
    #Elimino la columna Id y la columna diagnosisG pero creo una variable target donde almacena los datos de diagnosisG
    df = df.drop("Id", axis=1)

    columnsMinusDiag = df.columns
    columnsMinusDiag = columnsMinusDiag.drop("diagnosis")
    # Recorrer cada columna, y si el valor mínimo es 0 entonces incrementar 1 a toda la columna
    for col in columnsMinusDiag:
        minimo_actual = df[col].min()
        # Verificar si el mínimo es 0
        if minimo_actual == 0:
            df[col] = df[col] + 1
        # print("df col", df[col])

    num = 1
    for col in columnsMinusDiag:
        df[col] = num + df[col] / 100
        num = num + 1

    #aplicar one hot encoding a todas las columnas del df excepto de diagnosis
    target = pd.DataFrame()
    target["diagnosis"] = df["diagnosis"]
    df = df.drop("diagnosis", axis=1)
    df = pd.get_dummies(df, columns=columnsMinusDiag)
    df["diagnosis"] = target["diagnosis"]

    #incrementar el df muestreando aleatoriamente 10% de los datos donde diagnosis es 1
    dfTrue = df[df["diagnosis"].values == 1]
    dfFalse = df[df["diagnosis"].values == 0]
    #dfFalse = dfFalse.sample(frac=0.5)

    ##Guarda en un csv los datos de cada dataframe "dfTrue y dfFalse"
    #dfTrue.to_csv('dfTrue.csv', index=False)
    #dfFalse.to_csv('dfFalse.csv', index=False)

    dfTrue = pd.concat([dfTrue]*6)
    df = pd.concat([dfTrue, dfFalse])
    #resetear el indice del df
    df.reset_index(drop=True, inplace=True)

    # # imprimir value_counts de cada una de las columnas de selectedForTest donde targetSample es 1
    # for col in df.columns:
    #     print(df[col][df["diagnosis"] == 1].value_counts().sort_values())

    target = pd.DataFrame()
    target["diagnosis"] = df["diagnosis"]
    df = df.drop("diagnosis", axis=1)
    target["CRV"] = 0

    #obtener una muestra de 10% de los datos donde diagnosis es 1
    #dfTrue = dfTrue.sample(frac=0.1)
    return (df, target)


def evaluar(HM, i):
    global correlaciones, df, target
    #multiplicar todos los elementos del df por HM[i] y sumar el resultado de cada renglón para almacenarlo en target["CRV"]
    target["CRV"] = df.mul(HM[i]).sum(axis=1)

    correlaciones[i], p_value = pearsonr(target["CRV"], target["diagnosis"])

    return correlaciones[i]

def calcularTodasLasCorrelaciones(HM):
    global correlaciones, df, target
    for i in range(len(HM)):
        #multiplicar todos los elementos del df por HM[i] y sumar el resultado de cada renglón para almacenarlo en target["CRV"]
        target["CRV"] = df.mul(HM[i]).sum(axis=1)

        correlaciones[i], p_value = pearsonr(target["CRV"], target["diagnosis"])
        #un valor pequeño en p_value implica una correlación significativa ya sea positiva o negativa pero lejos del 0
    return correlaciones

# Función para generar una nueva armonía seleccionando valores de otros diccionarios
def Seleccionar_tono_aleatorio(HM, ind_correlacion_peor, pitch):
    # Elige aleatoriamente un valor del pitch actual de HM para la nueva armonía sin considerar al renglón ind_correlacion_peor
    #No es conveniente bloquear la selección ya que evita que sea seleccionado un pitch igual al de la peor solución aunque sea de otra arminía.
    #return random.choice(HM[:, pitch] != HM[ind_correlacion_peor][pitch])
    return random.choice(HM[:, pitch])

#Función para generar un tono aleatorio para una nueva armonía
def generar_tono_aleatorio(dictionarios, pitch):
    # Toma los limites de los valores a utilizar
    global  pitch_aleatorio, max_value_temp, min_value_temp, porcentaje_pitch_aleatorio_MIN_MAX
    # Se puede utilizar como max_value_temp el MAX_VAL o el valor máximo del pitch en los diccionarios
    #se puede utilizar como min_value_temp el MIN_VAL o el valor mínimo del pitch en los diccionarios
    if (random.uniform(0, 1) < porcentaje_pitch_aleatorio_MIN_MAX):
        max_value_temp =  MAX_VAL
        min_value_temp =  MIN_VAL
    else:
        max_value_temp = max(HM[:, pitch])
        min_value_temp = min(HM[:, pitch])

    # Generar un valor aleatorio entre 0 y 1, limitando los decimales a 4 digitos.
    valor_random = round(random.uniform(0, 1), 4)
    # Generar un valor aleatorio para el tono con la siguiente formula:
    pitch_aleatorio = (max_value_temp - min_value_temp) * valor_random + min_value_temp
    # Retornar el pitch_aleatorio redondeado sin decimales.
    return round(pitch_aleatorio,0)


# Función para ajustar el tono de los valores en un diccionario, teniendo en cuenta los valores máximos y mínimos de tono
def ajustar_tono(HM, ind_correlacion_peor, pitch):
    global pitch_ajustado, valor_random_temp, min_value_temp, max_value_temp
    # Obtener los valores de la columna maximos y minimos de un pitch
    min_value_temp = min(HM[:, pitch])
    max_value_temp = max(HM[:, pitch])

    # Generar un valor aleatorio entre -1 y 1, limitando los decimales a 4 digitos.
    valor_random_temp = round(random.uniform(-1, 1), 4)

    # Calcular el ajuste de tono basado en el valor aleatorio. Se multiplica por .3 para disminuir el movimiento de x
    pitch_ajustado = (max_value_temp - min_value_temp) * valor_random_temp * .3 + HM[ind_correlacion_peor][pitch]
    # Tomar el valor absoluto del ajuste de tono
    if pitch_ajustado < 0:
        pitch_ajustado = 0
    # Retornar el pitch_ajustado redondeado sin decimales
    return round(pitch_ajustado,0)


if __name__ == "__main__":
    # Establecer una semilla
    random.seed(42)
    np.random.seed(42)
    pd.np.random.seed(42)
    # Variables globales temporales
    pitch_aleatorio = 0
    pitch_ajustado = 0
    valor_random_temp = 0
    min_value_temp = 0
    max_value_temp = 0

    NUM_DICT = 11
    MIN_VAL = 0
    MAX_VAL = 20
    HMCR = 0.7
    PAR = 0.3
    iters_sin_mejora = 0
    intentos = 0
    MAX_ITERS_SIN_MEJORA = 10000
    correlaciones = np.zeros(NUM_DICT)
    porcentaje_pitch_aleatorio_MIN_MAX = 0.4

    # preprocesa el dataframe e incrementa el número de registros donde diagnosis es 1
    df, target = PreProcess('BCSC_risk_factors_153821.csv')

    #inicializa Harmony Memory
    HM = np.random.randint(MIN_VAL, MAX_VAL, size=(NUM_DICT, len(df.columns)))

    # Calcular las correlaciones iniciales de las armonías
    correlaciones = calcularTodasLasCorrelaciones(HM)
    bestCorrel = max(correlaciones)
    print("Correlaciones iniciales", correlaciones)
    print()


    while iters_sin_mejora < MAX_ITERS_SIN_MEJORA:
        iters_sin_mejora += 1
        # Encontrar la peor correlación
        # del arreglo de correlaciones, obtiene el índice de la peor correlación
        ind_correlacion_peor = correlaciones.argmin()

        #print("peor correlacion", ind_correlacion_peor)

        for pitch in range(HM.shape[1]):
            # Generar dos valores aleatorios entre 0 y 1
            valor_aleatorio1 = random.uniform(0, 1)
            valor_aleatorio2 = random.uniform(0, 1)

            if valor_aleatorio1 <= HMCR:
                #Seleccionar tono aleatorio.
                HM[ind_correlacion_peor][pitch] = Seleccionar_tono_aleatorio(HM, ind_correlacion_peor, pitch)
            else:
                # Genera un tono aleatorio para la nueva armonía
                HM[ind_correlacion_peor][pitch] = generar_tono_aleatorio(HM, pitch)
            if valor_aleatorio2 <= PAR:
                # Ajusta el tono de la nueva armonía generada
                HM[ind_correlacion_peor][pitch] = ajustar_tono(HM, ind_correlacion_peor, pitch)

        # Calcula la correlación de la nueva armonía generada
        correlaciones[ind_correlacion_peor] = evaluar(HM,ind_correlacion_peor)


        if bestCorrel < correlaciones[ind_correlacion_peor]:
            # Reiniciar el contador del número de iteraciones sin mejora
            bestCorrel = correlaciones[ind_correlacion_peor]
            print("Mejoró, BestCorrel = ", bestCorrel,"iterSinMejora = ", iters_sin_mejora)
        if ind_correlacion_peor != correlaciones.argmin():
            iters_sin_mejora = 0


        intentos += 1

    #Imprime el número total de intentos realizados
    print("Total Intentos:", intentos)
    print("Correlaciones finales", correlaciones)
    print()


    indexBestSol = correlaciones.argmax()
    result = list(zip(df.columns.values, HM[indexBestSol]))


    with open("Soluciones optimasv2.csv", "a", newline="") as f:
        #escribe en el archivo f la fecha y las variables globales de configuración

        f.write("Fecha: {0}\n".format(pd.Timestamp.now()))
        f.write("HMCR: {0}\n".format(HMCR))
        f.write("PAR: {0}\n".format(PAR))
        f.write("MAX_ITERS_SIN_MEJORA: {0}\n".format(MAX_ITERS_SIN_MEJORA))
        f.write("NUM_DICT: {0}\n".format(NUM_DICT))
        f.write("MIN_VAL: {0}\n".format(MIN_VAL))
        f.write("MAX_VAL: {0}\n".format(MAX_VAL))
        f.write("Total Intentos: {0}\n".format(intentos))
        f.write("Mejor correlación: {0}\n".format(bestCorrel))
        f.write("Correlaciones: {0}\n".format(correlaciones))
        f.write("porcentaje_pitch_aleatorio_MIN_MAX: {0}\n".format(porcentaje_pitch_aleatorio_MIN_MAX))
        #target["CRVBoolean"] = target["CRV"].apply(lambda x: 1 if x > 85 else 0)
        #f.write("Total de casos correctamente clasificados: {0}\n".format((target["CRVBoolean"] == target["diagnosis"]).sum()))
        target["CRV"] = df.mul(HM[indexBestSol]).sum(axis=1)
        inicio = int((np.average(target["CRV"]) + np.min(target["CRV"])) // 2)
        fin = int((np.average(target["CRV"]) + np.max(target["CRV"])) // 2)

        for limite in range(inicio, fin):
            target["CRVBoolean"] = target["CRV"].apply(lambda x: 1 if x > limite else 0)
            accuracy = (target["CRVBoolean"] == target["diagnosis"]).sum() / target.shape[0]
            precision = (target["CRVBoolean"].mul(target["diagnosis"])).sum() / (target["diagnosis"] == 1).sum()
            sensitivity = (target["CRVBoolean"].mul(target["diagnosis"])).sum() / (target["CRVBoolean"]).sum()
            specificity = (1 - target["CRVBoolean"]).mul(1-target["diagnosis"]).sum() / (target["CRVBoolean"] == 0).sum()
            f1 = 2*(precision*sensitivity)/(precision + sensitivity)

            f.write("accuracy: {0} para limite = {1}\t".format(accuracy,limite))
            f.write("precision: {0}\t".format(precision))
            f.write("Sensitivity: {0}\t".format(sensitivity))
            f.write("Specificity: {0}\t".format(specificity))
            f.write("f1: {0}\t".format(f1))

            TP = (target["CRVBoolean"].mul(target["diagnosis"])).sum()
            FP = ((1-target["CRVBoolean"]).mul(target["diagnosis"])).sum()
            FN = (target["CRVBoolean"].mul(1- target["diagnosis"])).sum()
            TN = (1 - target["CRVBoolean"]).mul(1-target["diagnosis"]).sum()

            f.write("TP: {0}\t".format(TP))
            f.write("FP: {0}\t".format(FP))
            f.write("FN: {0}\t".format(FN))
            f.write("TN: {0}\n".format(TN))

        writer = csv.writer(f)
        #writer.writerow(['Clave', 'Valor'])  # Escribir encabezados de columnas
        writer.writerows(result)
        f.write("\n")

    print('Los datos se han escrito en el archivo CSV.')

















