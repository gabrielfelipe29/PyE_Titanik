from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataframe = pd.read_csv('titanik.csv')

media_edades_por_genero = dataframe.groupby('gender')['age'].mean()
print("\nMedia de edad por género:")
print(media_edades_por_genero)

# Función para rellenar los valores faltantes con la media correspondiente
def completar_edades(row):
    if pd.isnull(row['age']):
        return media_edades_por_genero[row['gender']]
    else:
        return row['age']

# Aplicar la función al DataFrame
dataframe['age'] = dataframe.apply(completar_edades, axis=1)


moda = dataframe.groupby('gender')['age'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
print("\nModa: ", moda)

mediana = dataframe.groupby('gender')['age'].median()
print("\nMediana: ", mediana)

maximo = dataframe.groupby('gender')['age'].max()
minimo = dataframe.groupby('gender')['age'].min()

print(f"\nRango:  {maximo - minimo}")

desviacion_estandar=dataframe.groupby('gender')['age'].std()
print(f"\nDesviación estándar: {desviacion_estandar}")

varianza= dataframe.groupby('gender')['age'].var()
print(f"\nVarianza: {varianza}")


tasa_supervivencia_general = dataframe['survived'].mean()
print(f"\nTasa de supervivencia en general: {tasa_supervivencia_general:.2%}")

# Calcular la tasa de supervivencia por género
tasa_de_supervivencia_genero = dataframe.groupby('gender')['survived'].mean() * 100
print("\nTasa de supervivencia por género:")
print(tasa_de_supervivencia_genero)

print("\nNo sobrevivieron:")

# Filtrar los registros donde la columna 'survived' es igual a 0
survived_0 = dataframe[dataframe['survived'] == 0]

# Agrupar por género y contar la cantidad de 0 en la columna 'survived'
tasa_de_supervivencia_genero_0 = survived_0.groupby('gender').size()  * 100 / dataframe.groupby('gender').size()

# Mostrar el resultado
print(tasa_de_supervivencia_genero_0)


# Funcion para graficar los diagramas de caja
def diagrama_caja(datos, titulo, xlabel, ylabel):
    fig, ax = plt.subplots()

    
    box = ax.boxplot(datos, patch_artist=True, vert=True, widths = 0.6)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([1,2],["Superviviente", "No superviviente"])
        
    colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    i=0
    for xtick in ax.get_xticks():
        median = np.median(datos[i])
        q1 = np.percentile(datos[i], 25)
        q3 = np.percentile(datos[i], 75)
        ax.text(xtick, median, f'{median:.2f}',  horizontalalignment='left', va="top", size=7, color='k', weight='semibold', bbox=dict(facecolor='lightgray', pad=5))
        ax.text(xtick, q1, f'{q1:.2f}', horizontalalignment='left', va="bottom", size=7,color='k', weight='semibold', bbox=dict(facecolor='lightgray', pad=5))
        ax.text(xtick, q3, f'{q3:.2f}', horizontalalignment='left', va="top", size=7, color='k', weight='semibold', bbox=dict(facecolor='lightgray', pad=5))
        i+=1
    plt.savefig(f'{titulo}.png')



# Funcion para graficar el histograma
def histograma(datos, titulo, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.hist(datos, color='blue', bins=20, edgecolor='black')
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'Histograma {titulo}.png')



edades_supervivientes = dataframe[dataframe['survived'] == 1]['age'].dropna()  # Filtra por supervivientes y elimina NaN
edades_no_supervivientes = dataframe[dataframe['survived'] == 0]['age'].dropna()  # Filtra por supervivientes y elimina NaN

# No supervivientes
#diagrama_caja([edades_supervivientes, edades_no_supervivientes], "Cajas supervivientes", "Categoría", "Edad")



#tasa_supervivencia_mujeres = survival_rate_by_gender['Female']  

# Filtrar los registros donde la columna 'p_class' es igual a 1
primera_clase = dataframe[dataframe['p_class'] == 1]['age']
segunda_clase = dataframe[dataframe['p_class'] == 2]['age']
tercera_clase = dataframe[dataframe['p_class'] == 3]['age']
todo = dataframe['age']


# Convertir la Serie resultante a una lista
lista_edades_primera_clase = primera_clase.tolist()
lista_edades_segunda_clase = segunda_clase.tolist()
lista_edades_tercera_clase = tercera_clase.tolist()

# Hacemos el histograma para primera
histograma(lista_edades_primera_clase, "Primera clase", "Edades", "Cantidad" )

# Hacemos el histograma para segunda
histograma(lista_edades_segunda_clase, "Segunda clase", "Edades", "Cantidad" )

# Hacemos el histograma para tercera
histograma(lista_edades_tercera_clase, "Tercera clase", "Edades", "Cantidad" )

# Todas las edades
histograma(todo, "Todo", "Edades", "Cantidad" )

# Ejercicio 1 - Intervalo de confianza del 95% para la edad promedio
ic_todos=stats.t.interval(confidence=0.05, loc=dataframe['age'].mean(), scale=dataframe['age'].std(), df=(len(dataframe) - 1))
print("\nIntervalo de confianza t: ")
print(ic_todos)


# Ejercicio 2 - Prueba de hipotesis para las mujeres con 95% de confianza
# H0: mu_mujeres=56
# H1: mu_mujeres>56

print("\nPrueba de hipotesis edad mujeres:")
t_statistic, p_value= stats.ttest_1samp(a=dataframe[dataframe['gender']=='female']['age'], popmean=56, alternative='greater')
print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")

# T-Statistic: 0.7395675473549336
# p-value: 0.2298726687561159
# Como p-value=0.22 > alfa=0.05, no se rechaza H0.
# El promedio de edades de las mujeres no difiere significativamente de los 56 años.


# Prueba de hipotesis para los hombres con 95% de confianza
# H0: mu_hombres=56
# H1: mu_hombres>56

print("\nPrueba de hipotesis edad hombres:")

t_statistic, p_value= stats.ttest_1samp(a=dataframe[dataframe['gender']=='male']['age'], popmean=56, alternative='greater')

print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")

# T-Statistic: 3.1950677480438108
# p-value: 0.0007215042458834663
# como p-value=0.0007 < alfa=0.05, entonces se rechaza H0.
# El promedio de edades de los hombres es mayor a 56 años


# Ejercicio 3 - Tasa de supervivencia de hombres y mujeres
# 
# H0: Tasa_hombres=Tasa_mujeres
# H1: Tasa_hombres!=Tasa_mujeres

print("\nPrueba de hipotesis tasa supervivencia hombres y mujeres:")

ds_hombres=dataframe[(dataframe['gender'] == 'male')]

ds_mujeres=dataframe[(dataframe['gender']== 'female')]

#print(f"Tasas supervivencia H: {ds_hombres['survived'].mean()} y M {ds_mujeres['survived'].mean()}")

t_statistic, p_value = stats.ttest_ind(a=ds_hombres['survived'], b=ds_mujeres['survived'], alternative="two-sided", equal_var=True)
print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")

# T-Statistic: -6.8840586510873285
# p-value: 7.849335669088787e-12
# Como p-value=0.000000000007.8493 < alfa=0.01, se rechaza H0. Es decir, las tasas de supervivencia difieren significativamente.

print("\nPrueba de hipotesis tasa supervivencia primera clase y segunda clase:")

clase_1=dataframe[(dataframe['p_class'] == 1)]

clase_2=dataframe[(dataframe['p_class'] == 2)]

clase_3=dataframe[(dataframe['p_class'] == 3)]


# H0: Tasa_primera=Tasa_segunda
# H1: Tasa_primera!=Tasa_segunda

t_statistic, p_value = stats.ttest_ind(a=clase_1['survived'], b=clase_2['survived'], alternative="two-sided", equal_var=True)
print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")

# T-Statistic: 26.691766153419707
# p-value: 6.251096933708115e-123
# Como p-value=6.251096933708115e-123 < alfa=0.01, entonces se rechaza H0.
# Las tasas de supervivencia en primera y segunda clase difieren significativamente.


print("\nPrueba de hipotesis tasa supervivencia primera clase y tercera clase:")

# H0: Tasa_primera=Tasa_tercera
# H1: Tasa_primera!=Tasa_tercera

t_statistic, p_value = stats.ttest_ind(a=clase_1['survived'], b=clase_3['survived'], alternative="two-sided", equal_var=True)
print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")

# T-Statistic: 42.21076330438675
# p-value: 9.026447143885264e-234
# Como p-value=9.026447143885264e-234 < alfa=0.01, entonces se rechaza H0. 
# Las tasas de supervivencia en primera y tercera clase difieren significativamente

print("\nPrueba de hipotesis tasa supervivencia segunda clase y tercera clase:")

# H0: Tasa_segunda=Tasa_tercera
# H1: Tasa_segunda!=Tasa_tercera

t_statistic, p_value = stats.ttest_ind(a=clase_2['survived'], b=clase_3['survived'], alternative="two-sided", equal_var=True)
print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")

# T-Statistic: 7.767761626497189
# p-value: 1.4419985410591502e-14
# Como p-value = 1.4419985410591502e-14 < alfa=0.01, entonces se rechaza H0.
# Las tasas de supervivencia en segunda y tercera clase difieren significativamente


# Ejercicio 4 - Edad de hombres y mujeres
# 
# H0: Edad_mujeres=Edad_hombres
# H1: Edad_mujeres<Edad_hombres

print("\nPrueba de hipotesis edad hombres y mujeres:")

edad_hombres=dataframe[dataframe['gender']=='male']['age']
edad_mujeres=dataframe[dataframe['gender']=='female']['age']

t_statistic, p_value = stats.ttest_ind(a=edad_mujeres, b=edad_hombres, equal_var=True, alternative='less')
print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")

# T-Statistic: -1.7051508685967423
# p-value: 0.04416398359724209

# Como p-value = 0.044 < alfa = 0.05, se rechaza H0.
# Las mujeres son más jovenes que los hombres. 