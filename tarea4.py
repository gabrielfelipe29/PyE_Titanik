from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Formato datos: passenger_id,survived,p_class,title_name,full_name,gender,age,siblings,parents,embarked

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



print("\nDataFrame después de reemplazar los valores faltantes:")
print(dataframe.head(20))

moda = dataframe.groupby('gender')['age'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
print("Moda: ", moda)

mediana = dataframe.groupby('gender')['age'].median()
print("Mediana: ", mediana)

maximo = dataframe.groupby('gender')['age'].max()
minimo = dataframe.groupby('gender')['age'].min()

print(f"Rango: {minimo} a {maximo}")

desviacion_estandar=dataframe.groupby('gender')['age'].std()
print(f"Desviación estándar: {desviacion_estandar}")

varianza= dataframe.groupby('gender')['age'].var()
print(f"Varianza: {varianza}")


tasa_supervivencia_general = dataframe['survived'].mean()
print(f"Tasa de supervivencia en general: {tasa_supervivencia_general:.2%}")

# Calcular la tasa de supervivencia por género
tasa_de_supervivencia_genero = dataframe.groupby('gender')['survived'].mean() * 100
print("Tasa de supervivencia por género:")
print(tasa_de_supervivencia_genero)

print("No sobrevivieron")

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

# Intervalo de confianza del 95% para la edad promedio
# estaria bien esto o hay que usar stats.t.interval?
ic_todos=stats.norm.interval(confidence=0.95, loc=dataframe['age'].mean(), scale=dataframe['age'].std())
print("Intervalo de confianza: ")
print(ic_todos)


# Ejercicio 2 - REVISAR
# Prueba de hipotesis para las mujeres con 95% de confianza
# H0: mu_mujeres=56
# H1: mu_mujeres!=56

print("Prueba de hipotesis edad mujeres:")
t_statistic, p_value= stats.ttest_1samp(a=dataframe[dataframe['gender']=='female']['age'], popmean=56)

print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")

# T-Statistic: 0.7395675473549336
# p-value: 0.4597453375122318
# Como p-value=0.46 > alfa=0.05, no se rechaza H0.
# El promedio de edades de las mujeres no difiere significativamente de los 56 años.


# Prueba de hipotesis para los hombres con 95% de confianza
# H0: mu_hombres=56
# H1: mu_hombres!=56

print("Prueba de hipotesis edad hombres:")

t_statistic, p_value= stats.ttest_1samp(a=dataframe[dataframe['gender']=='male']['age'], popmean=56)

print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")

print(dataframe[dataframe['gender']=='male']['age'].mean())


# Ejercicio 3 - Tasa de supervivencia de hombres y mujeres
# 
# H0: mu_hombres-mu_mujeres=0
# H1: mu_hombres-mu_mujeres!=0

print("Prueba de hipotesis edad hombres y mujeres:")

ds_hombres=tasa_de_supervivencia_genero['male'].std()
ds_mujeres=tasa_de_supervivencia_genero['female'].std()

if((ds_hombres/ds_mujeres)>=2 or (ds_mujeres/ds_hombres)>=2):
    var_equal=False

t_statistic, p_value = stats.ttest_ind(a=dataframe[dataframe['gender']=='male']['age'], b=dataframe[dataframe['gender']=='female']['age'],equal_var=var_equal)
print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")

#stats.binomtest()




# Ejercicio 4 - Edad de hombres y mujeres
# 
# H0: mu_hombres-mu_mujeres=0
# H1: mu_hombres-mu_mujeres!=0

print("Prueba de hipotesis edad hombres y mujeres:")

ds_hombres=dataframe[dataframe['gender']=='male']['age'].std()
ds_mujeres=dataframe[dataframe['gender']=='female']['age'].std()

if((ds_hombres/ds_mujeres)>=2 or (ds_mujeres/ds_hombres)>=2):
    var_equal=False

t_statistic, p_value = stats.ttest_ind(a=dataframe[dataframe['gender']=='male']['age'], b=dataframe[dataframe['gender']=='female']['age'],equal_var=var_equal)
print(f"T-Statistic: {t_statistic}")
print(f"p-value: {p_value}")