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
tasa_de_supervivencia_genero = dataframe.groupby('gender')['survived'].mean()
print("Tasa de supervivencia por género:")
print(tasa_de_supervivencia_genero)



