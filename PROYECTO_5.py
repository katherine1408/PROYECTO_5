

# # ¿Cuál es la mejor tarifa?
# 
# Trabajas como analista para el operador de telecomunicaciones Megaline. La empresa ofrece a sus clientes dos tarifas de prepago, Surf y Ultimate. El departamento comercial quiere saber cuál de las tarifas genera más ingresos para poder ajustar el presupuesto de publicidad.
# 
# Vas a realizar un análisis preliminar de las tarifas basado en una selección de clientes relativamente pequeña. Tendrás los datos de 500 clientes de Megaline: quiénes son los clientes, de dónde son, qué tarifa usan, así como la cantidad de llamadas que hicieron y los mensajes de texto que enviaron en 2018. Tu trabajo es analizar el comportamiento de los clientes y determinar qué tarifa de prepago genera más ingresos.

# ## Inicialización

# In[1]:


# Cargar todas las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy.stats import ttest_ind, levene


# ## Cargar datos

# In[2]:


### Carga los archivos de datos en diferentes DataFrames

llamadas_df= pd.read_csv('megaline_calls.csv')
internet_df= pd.read_csv('megaline_internet.csv')
mensajes_df= pd.read_csv('megaline_messages.csv')
planes_df  = pd.read_csv('megaline_plans.csv')
user_df    = pd.read_csv('megaline_users.csv')


# ## Preparar los datos

# ## Tarifas

# In[3]:


# Imprime la información general/resumida sobre el DataFrame de las tarifas
planes_df.info()


# In[4]:


# Imprime una muestra de los datos para las tarifas
planes_df.head()



# ## Corregir datos

# In[5]:


print(planes_df.isna().sum())


# In[6]:


print(planes_df.duplicated().sum())


# ## Enriquecer los datos

# In[7]:


planes_df.rename(columns={'plan_name':'plan_id'},inplace=True)
planes_df.head()




# In[8]:


planes_df['mb_per_month_included'] = np.ceil(planes_df['mb_per_month_included'] / 1024)
planes_df.head()


# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# 
# Se aplico las correccion correspondientes de acuerdo a lo sugerido, se covirtió los mb a gb
# 
#      
# </div>

# ## Usuarios/as

# In[9]:


# Imprime la información general/resumida sobre el DataFrame de usuarios
user_df.info()


# In[10]:


# Imprime una muestra de datos para usuarios
user_df.head()


# In[11]:


user_df.describe()


# In[12]:


print(planes_df.duplicated().sum())



# 
# Observamos que existen valores nulos en la columna churn_date (fecha en la que el usuario dejo de usar el usuario), está información es razonable debido a que contiene la información de usuarios aún activos , por lo que conservaremos los valores nulos, adicionalmente las columnas'reg_date'y 'churn_date' se encuentran en tipo objeto por lo que es necesario modificar sus valores a el formato Datatime.

# In[13]:


user_df.rename(columns={'plan':'plan_id'},inplace=True)


# ### Enriquecer los datos

# In[14]:


user_df['reg_date']=pd.to_datetime(user_df['reg_date'],format='%Y-%m-%d')
user_df['reg_mes']=pd.DatetimeIndex(user_df['reg_date']).month
user_df['churn_date']=pd.to_datetime(user_df['churn_date'],format='%Y-%m-%d')
user_df['churn_mes']=pd.DatetimeIndex(user_df['churn_date']).month

user_df.info()


# ## Llamadas

# In[15]:


# Imprime la información general/resumida sobre el DataFrame de las llamadas
llamadas_df.info()


# In[16]:


# Imprime una muestra de datos para las llamadas
llamadas_df.head()


# In[17]:


llamadas_df.describe()


# [Corrige los problemas obvios con los datos basándote en las observaciones iniciales.]

# In[18]:


print(llamadas_df.isna().sum())
print(llamadas_df.duplicated(subset=['id','user_id']).sum())



# ### Enriquecer los datos

# In[19]:


llamadas_df.rename(columns={'id':'call_id'},inplace=True)
llamadas_df['call_date']=pd.to_datetime(llamadas_df['call_date'],format='%Y-%m-%d')
llamadas_df['mes']=pd.DatetimeIndex(llamadas_df['call_date']).month
llamadas_df['duracion_redondeada']=np.ceil(llamadas_df['duration'])
llamadas_df.head()


# In[20]:


(len(llamadas_df.query('duration == 0'))/len(llamadas_df))*100


# In[21]:


llamadas_df=llamadas_df[llamadas_df['duration']>0]


# In[22]:


len(llamadas_df.query('duration == 0'))



# ## Mensajes

# In[23]:


# Imprime la información general/resumida sobre el DataFrame de los mensajes
mensajes_df.info()


# In[24]:


# Imprime una muestra de datos para los mensajes
mensajes_df.head()


## Corregir los datos

# In[25]:


print(mensajes_df.isna().sum())
print(mensajes_df.duplicated().sum())


# ### Enriquecer los datos

# In[26]:


mensajes_df.rename(columns={'id':'message_id'},inplace=True)
mensajes_df['message_date']=pd.to_datetime(mensajes_df['message_date'],format='%Y-%m-%d')
mensajes_df['mes']=pd.DatetimeIndex(mensajes_df['message_date']).month
mensajes_df.head()


# ## Internet

# In[27]:


# Imprime la información general/resumida sobre el DataFrame de internet
internet_df.info()


# In[28]:


# Imprime una muestra de datos para el tráfico de internet
internet_df.head()


# In[29]:


internet_df.describe()



# ### Corregir los datos

# In[30]:


print(internet_df.isna().sum())
print(internet_df.duplicated(subset=['id','user_id']).sum())



# ### Enriquecer los datos

# In[31]:


internet_df.rename(columns={'id':'session_id'},inplace=True)
internet_df['session_date']=pd.to_datetime(internet_df['session_date'],format='%Y-%m-%d')
internet_df['mes']=pd.DatetimeIndex(internet_df['session_date']).month
internet_df.head()


# In[32]:


(len(internet_df.query('mb_used == 0'))/len(internet_df))*100


# ## Estudiar las condiciones de las tarifas

# In[33]:


# Imprime las condiciones de la tarifa y asegúrate de que te quedan claras

surf_pago_mensual=20
surf_minutos_mensual=500
surf_mensajes_mensual=50
surf_datos_mensual=15
surf_minutos_extra=0.03
surf_mensajes_extra=0.03
surf_datos_extra=10

ultimate_pago_mensual=70
ultimate_minutos_mensual=3000
ultimate_mensajes_mensual=1000
ultimate_datos_mensual=30
ultimate_minutos_extra=0.01
ultimate_mensajes_extra=0.01
ultimate_datos_extra=7


# ## Agregar datos por usuario
# 
# 



# In[34]:


# Calcula el número de llamadas hechas por cada usuario al mes. Guarda el resultado:
# Calcula la cantidad de minutos usados por cada usuario al mes. Guarda el resultado:
llamadas_analytics=llamadas_df.pivot_table(index=['user_id','mes'],values='duracion_redondeada',aggfunc=['sum','count'])
llamadas_analytics.columns=['minutos_usados','llamadas_hechas']
llamadas_analytics=llamadas_analytics.reset_index()
llamadas_analytics


# In[35]:


# Calcula el número de mensajes enviados por cada usuario al mes. Guarda el resultado.
mensajes_analytics=mensajes_df.pivot_table(index=['user_id','mes'],values='message_id',aggfunc='count')
mensajes_analytics.columns=['cantidad_de_mensajes']
mensajes_analytics=mensajes_analytics.reset_index()
mensajes_analytics


# In[36]:


# Calcula el volumen del tráfico de Internet usado por cada usuario al mes. Guarda el resultado.
internet_analytics=internet_df.pivot_table(index=['user_id','mes'],values='mb_used',aggfunc='sum')
internet_analytics.columns=['total_de_datos_usados']
internet_analytics=internet_analytics.reset_index()

internet_analytics['total_de_datos_usados_redondeados'] = np.ceil(internet_analytics['total_de_datos_usados'] / 1024)
internet_analytics


#       
# </div>

# In[37]:


# Fusiona los datos de llamadas, minutos, mensajes e Internet con base en user_id y month

llamadas_mensajes= llamadas_analytics.merge(mensajes_analytics, on = ['user_id','mes'],how='outer')
tabla_de_consumo = llamadas_mensajes.merge(internet_analytics, on = ['user_id','mes'],how='outer')
tabla_de_consumo




# In[38]:


# Reemplazando valores nulos
tabla_de_consumo.fillna(0,inplace=True)
tabla_de_consumo


# In[39]:


# Añade la información de la tarifa

plan_analytics = user_df[['user_id','plan_id', 'city']]

tabla_de_consumo_planes = tabla_de_consumo.merge(plan_analytics,on=['user_id'],how='outer')
tabla_de_consumo_planes.fillna(0,inplace=True)
tabla_de_consumo_planes.head()


# In[40]:


# Calcula el ingreso mensual para cada usuario

def llamadas_por_cobrar (fila):
    plan=fila['plan_id']
    minutos_usados = fila['minutos_usados']
    
    minutos_cobrables = 0
    
    if plan == "surf":
        if minutos_usados > surf_minutos_mensual:
            minutos_cobrables= minutos_usados - surf_minutos_mensual
    elif plan == 'ultimate':
        if minutos_usados > ultimate_minutos_mensual:
            minutos_cobrables= minutos_usados - ultimate_minutos_mensual
        
    return minutos_cobrables   
    


# In[41]:


tabla_de_consumo_planes['minutos_cobrables'] = tabla_de_consumo_planes.apply(llamadas_por_cobrar, axis=1)


# In[42]:


# Definir los costos por minutos excedentes
def costo_llamadas(fila):
    plan = fila['plan_id']
    minutos_cobrables = fila['minutos_cobrables']
    
    if plan == 'surf':
        return minutos_cobrables * surf_minutos_extra
    elif plan == 'ultimate':
        return minutos_cobrables * ultimate_minutos_extra
    return 0

tabla_de_consumo_planes['costo_llamadas'] = tabla_de_consumo_planes.apply(costo_llamadas, axis=1)


# In[43]:


def mensajes_por_cobrar(fila):
    plan = fila['plan_id']
    mensajes_usados = fila['cantidad_de_mensajes']
    
    mensajes_cobrables = 0
    
    if plan == "surf":
        if mensajes_usados > surf_mensajes_mensual:
            mensajes_cobrables = mensajes_usados - surf_mensajes_mensual
    elif plan == 'ultimate':
        if mensajes_usados > ultimate_mensajes_mensual:
            mensajes_cobrables = mensajes_usados - ultimate_mensajes_mensual
    
    return mensajes_cobrables

tabla_de_consumo_planes['mensajes_cobrables'] = tabla_de_consumo_planes.apply(mensajes_por_cobrar, axis=1)

def costo_mensajes(fila):
    plan = fila['plan_id']
    mensajes_cobrables = fila['mensajes_cobrables']
    
    if plan == 'surf':
        return mensajes_cobrables * surf_mensajes_extra
    elif plan == 'ultimate':
        return mensajes_cobrables * ultimate_mensajes_extra
    return 0

tabla_de_consumo_planes['costo_mensajes'] = tabla_de_consumo_planes.apply(costo_mensajes, axis=1)


# In[44]:


def datos_por_cobrar(fila):
    plan = fila['plan_id']
    datos_usados = fila['total_de_datos_usados_redondeados']
    
    datos_cobrables = 0
    
    if plan == "surf":
        if datos_usados > surf_datos_mensual:
            datos_cobrables = datos_usados - surf_datos_mensual
    elif plan == 'ultimate':
        if datos_usados > ultimate_datos_mensual:
            datos_cobrables = datos_usados - ultimate_datos_mensual
    
    return datos_cobrables

tabla_de_consumo_planes['datos_cobrables'] = tabla_de_consumo_planes.apply(datos_por_cobrar, axis=1)

def costo_datos(fila):
    plan = fila['plan_id']
    datos_cobrables = fila['datos_cobrables']
    
    if plan == 'surf':
        return datos_cobrables * surf_datos_extra
    elif plan == 'ultimate':
        return datos_cobrables * ultimate_datos_extra
    return 0

tabla_de_consumo_planes['costo_datos'] = tabla_de_consumo_planes.apply(costo_datos, axis=1)


# In[45]:


def costo_mensual_base(fila):
    plan = fila['plan_id']
    if plan == 'surf':
        return surf_pago_mensual
    elif plan == 'ultimate':
        return ultimate_pago_mensual
    return 0

tabla_de_consumo_planes['costo_base'] = tabla_de_consumo_planes.apply(costo_mensual_base, axis=1)

# Ingreso total por usuario
tabla_de_consumo_planes['ingreso_total'] = (
    tabla_de_consumo_planes['costo_llamadas'] +
    tabla_de_consumo_planes['costo_mensajes'] +
    tabla_de_consumo_planes['costo_datos'] +
    tabla_de_consumo_planes['costo_base']
)



# ## Estudia el comportamiento de usuario

# ### Llamadas

# In[46]:


# Compara la duración promedio de llamadas por cada plan y por cada mes. Traza un gráfico de barras para visualizarla.

llamadas_promedio_por_mes = tabla_de_consumo_planes.groupby(['mes', 'plan_id'])['minutos_usados'].mean().unstack()
llamadas_promedio_por_mes


# In[47]:


llamadas_promedio_por_mes.plot(kind='bar', figsize=(10, 6))
plt.title('Duración Promedia de Llamadas por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Duración Promedia (minutos)')
plt.show()


# In[48]:


# Compara el número de minutos mensuales que necesitan los usuarios de cada plan. Traza un histograma.

numero_minutos_mensuales=tabla_de_consumo_planes.pivot_table(index=['user_id','plan_id'],values='minutos_usados',aggfunc='sum')
numero_minutos_mensuales = numero_minutos_mensuales.reset_index()
numero_minutos_mensuales


# In[49]:


minutos_plan_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['minutos_usados']
minutos_plan_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['minutos_usados']

# Crear el histograma
plt.hist(minutos_plan_surf, alpha=0.5, label='Plan Surf', bins=10)
plt.hist(minutos_plan_ultimate, alpha=0.5, label='Plan Ultimate', bins=10)


plt.xlabel('Minutos Mensuales')
plt.ylabel('Frecuencia')
plt.title('Histograma del Número de Minutos Mensuales por Plan')
plt.legend()

plt.show()




# In[50]:


# Calcula la media y la varianza de la duración mensual de llamadas.

#Plan Surf:
media_llamadas_surf= tabla_de_consumo_planes.query('plan_id=="surf"')['minutos_usados'].mean()
variacion_llamadas_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['minutos_usados'].var()
moda_llamadas_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['minutos_usados'].mode()
mediana_llamadas_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['minutos_usados'].median()
desviacion_estandar_llamadas_suf = tabla_de_consumo_planes.query('plan_id=="surf"')['minutos_usados'].std()
print(f'La media de duración  de las llamadas del plan surf es de: {media_llamadas_surf}')
print(f'La varianza de la duración de las llamadas del plan surf es de:{variacion_llamadas_surf}')
print(f'La moda de la duración de las llamadas del plan surf es de:{moda_llamadas_surf}')
print(f'La mediana de la duración de las llamadas del plan surf es de:{mediana_llamadas_surf}')
print(f'La desviacion estandar de la duración de las llamadas del plan surf es de:{desviacion_estandar_llamadas_suf}')


# In[51]:


#Plan Ultimate

media_llamadas_ultimate= tabla_de_consumo_planes.query('plan_id=="ultimate"')['minutos_usados'].mean()
variacion_llamadas_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['minutos_usados'].var()
moda_llamadas_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['minutos_usados'].mode()
mediana_llamadas_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['minutos_usados'].median()
desviacion_estandar_llamadas_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['minutos_usados'].std()

print(f'La media de duración de las llamadas del plan ultimate es de: {media_llamadas_ultimate}')
print(f'La varianza de la duración de las llamadas del plan ultimate es de:{variacion_llamadas_ultimate}')
print(f'La moda de la duración de las llamadas del plan ultimate es de:{moda_llamadas_ultimate}')
print(f'La mediana de la duración de las llamadas del plan ultimate es de:{mediana_llamadas_ultimate}')
print(f'La desviacion estandar de la duración de las llamadas del plan ultimate es de:{desviacion_estandar_llamadas_ultimate}')


# In[52]:


# Traza un diagrama de caja para visualizar la distribución de la duración mensual de llamadas

plt.figure(figsize=(10, 6))
tabla_de_consumo_planes.boxplot(column='minutos_usados', by='mes')
plt.title('Distribución de Duración Mensual de Llamadas')
plt.xlabel('Mes')
plt.ylabel('Duración de Llamadas')
plt.suptitle("")
plt.show()



# ### Mensajes

# In[53]:


# Comprara el número de mensajes que tienden a enviar cada mes los usuarios de cada plan
mensajes_totales_por_mes = tabla_de_consumo_planes.groupby(['mes', 'plan_id'])['cantidad_de_mensajes'].sum().unstack()
mensajes_totales_por_mes




# In[54]:


mediana_mensajes_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['cantidad_de_mensajes'].median()
moda_mensual_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['cantidad_de_mensajes'].mode()
desviacion_estandar_mensajes_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['cantidad_de_mensajes'].std()

print(f'La mediana de la cantidad de mensajes del plan surf es de:{mediana_mensajes_surf}')
print(f'La moda de la cantidad de mensajes del plan surf es de:{moda_mensual_surf}')
print(f'La desviacion estandar de la cantidad de mensajes del plan surf es de:{desviacion_estandar_mensajes_surf}')


# In[55]:


mediana_mensajes_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['cantidad_de_mensajes'].median()
moda_mensajes_ultimate=tabla_de_consumo_planes.query('plan_id=="ultimate"')['cantidad_de_mensajes'].mode()
desviacion_estandar_mensajes_ultimate=tabla_de_consumo_planes.query('plan_id=="ultimate"')['cantidad_de_mensajes'].std()

print(f'La mediana de la cantidad de mensajes del plan ultimate es de:{mediana_mensajes_ultimate}')
print(f'La moda  de la cantidad de mensajes del plan ultimate es de:{moda_mensajes_ultimate}')
print(f'La desviacion estandar de la cantidad de mensajes del plan ultimate es de:{desviacion_estandar_mensajes_ultimate}')


# In[56]:


mensajes_totales_por_mes.plot(kind='bar', figsize=(10, 6))
plt.title('Numero de mensajes por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Cantidad de usuarios')
plt.show()

# In[57]:


cantidad_mensajes_plan_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['cantidad_de_mensajes']
cantidad_mensajes_plan_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['cantidad_de_mensajes']

# Crear el histograma
plt.hist(cantidad_mensajes_plan_surf, alpha=0.5, label='Plan Surf', bins=10)
plt.hist(cantidad_mensajes_plan_ultimate, alpha=0.5, label='Plan Ultimate', bins=10)


plt.xlabel('Mensajes Mensuales')
plt.ylabel('Frecuencia')
plt.title('Número de Mensajes Mensuales por Plan')
plt.legend()

plt.show()


# In[58]:


plt.figure(figsize=(10, 6))
tabla_de_consumo_planes.boxplot(column='cantidad_de_mensajes', by='mes')
plt.title('Numero de mensajes por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Numero de mensajes')
plt.suptitle("")
plt.show()



# ### Internet

# In[59]:


datos_totales_por_plan = tabla_de_consumo_planes.groupby(['mes', 'plan_id'])['total_de_datos_usados_redondeados'].mean().unstack()
datos_totales_por_plan




# In[60]:


mediana_datos_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['total_de_datos_usados_redondeados'].median()
moda_datos_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['total_de_datos_usados_redondeados'].mode()
desviacion_estandar_datos_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['total_de_datos_usados_redondeados'].std()

print(f'La mediana de la cantidad de ddatos del plan surf es de:{mediana_datos_surf}')
print(f'La moda de la cantidad de datos del plan surf es de:{moda_datos_surf}')
print(f'La desviacion estandar de la cantidad de datos del plan surf es de:{desviacion_estandar_datos_surf}')


# In[61]:


mediana_datos_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['total_de_datos_usados_redondeados'].median()
moda_datos_ulltimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['total_de_datos_usados_redondeados'].mode()
desviacion_estandar_datos_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['total_de_datos_usados_redondeados'].std()

print(f'La mediana de la cantidad de datos del plan ultimate es de:{mediana_datos_ultimate}')
print(f'La moda  de la cantidad de datos del plan ultimate es de:{moda_datos_ulltimate}')
print(f'La desviacion estandar de la cantidad de datos del plan ultimate es de:{desviacion_estandar_datos_ultimate}')


# In[62]:


datos_totales_por_plan.plot(kind='bar', figsize=(10, 6))
plt.title('Datos utilizados por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Datos (Mb)')
plt.show()




# In[63]:


cantidad_datos_plan_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['total_de_datos_usados_redondeados']
cantidad_datos_plan_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['total_de_datos_usados_redondeados']

# Crear el histograma
plt.hist(cantidad_mensajes_plan_surf, alpha=0.5, label='Plan Surf', bins=90)
plt.hist(cantidad_mensajes_plan_ultimate, alpha=0.5, label='Plan Ultimate', bins=90)


plt.xlabel('Datos Mensuales')
plt.ylabel('Frecuencia')
plt.title('Datos Mensuales por Plan')
plt.legend()

plt.show()


# In[64]:


plt.figure(figsize=(10, 6))
tabla_de_consumo_planes.boxplot(column='total_de_datos_usados_redondeados', by='mes')
plt.title('Numero de mensajes por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Datos')
plt.suptitle("")
plt.show()




# ## Ingreso

# In[78]:


# Agrupar los ingresos por plan
ingresos_por_mes_plan = tabla_de_consumo_planes.groupby('plan_id',)['ingreso_total'].agg(['mean','sum', 'sum','max','min'])

# Mostrar resultados
print(ingresos_por_mes_plan)


# In[79]:


ingresos_por_plan = tabla_de_consumo_planes.groupby(['mes', 'plan_id'])['ingreso_total'].sum().unstack()

ingresos_por_plan.plot(kind='bar', figsize=(10, 6))
plt.title('Ingresos por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Ingresos ($))')
plt.show()


# In[67]:


ingresos_plan_surf = tabla_de_consumo_planes.query('plan_id=="surf"')['ingreso_total']
ingresos_plan_ultimate = tabla_de_consumo_planes.query('plan_id=="ultimate"')['ingreso_total']

# Crear el histograma
plt.hist(cantidad_mensajes_plan_surf, alpha=0.5, label='Plan Surf', bins=30)
plt.hist(cantidad_mensajes_plan_ultimate, alpha=0.5, label='Plan Ultimate', bins=30)


plt.xlabel('Ingresos Mensuales')
plt.ylabel('Frecuencia')
plt.title('Ingresos Mensuales por Plan')
plt.legend()

plt.show()


# In[68]:


plt.figure(figsize=(10, 6))
tabla_de_consumo_planes.boxplot(column='ingreso_total', by='mes')
plt.title('Ingreso por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Ingresos')
plt.suptitle("")
plt.show()


# Después del análisis, podemos concluir que hay una clara diferencia en los ingresos entre los planes "surf" y "ultimate". 
# Los clientes que utilizan el plan "surf" generan más ingresos , teniendo en cuenta que el promedio de usuarios con plan "Ultimate" tienen una tarifa de 70 usd ,esto sugiere que puede haber diferencias en el uso realizado por clientes suscritos a cada uno de estos planes, que podría estar relacionado a los excedentes de consumo realizados por los usuarios del plan ultimate, como lo muestran las gráficas a pesar de tener un plan base menor de 20 usd, se puede visualizar que se exceden en sus consumos.      

# 

# In[69]:


llamadas_promedio_por_mes = tabla_de_consumo_planes.groupby(['mes', 'plan_id'])['minutos_usados'].sum().unstack()
llamadas_promedio_por_mes.plot(kind='line', figsize=(10, 6))
plt.title('Duración Promedia de Llamadas por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Duración Promedia (minutos)')
plt.show()


# In[70]:


mensajes_totales_por_mes = tabla_de_consumo_planes.groupby(['mes', 'plan_id'])['cantidad_de_mensajes'].mean().unstack()
mensajes_totales_por_mes.plot(kind='line', figsize=(10, 6))
plt.title('Numero de mensajes por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Cantidad de usuarios')
plt.show()


# In[71]:


datos_totales_por_plan = tabla_de_consumo_planes.groupby(['mes', 'plan_id'])['total_de_datos_usados_redondeados'].sum().unstack()

datos_totales_por_plan.plot(kind='line', figsize=(10, 6))
plt.title('Datos utilizados por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Datos (Mb)')
plt.show()


# In[72]:


ingresos_por_plan = tabla_de_consumo_planes.groupby(['mes', 'plan_id'])['ingreso_total'].sum().unstack()

ingresos_por_plan.plot(kind='line', figsize=(10, 6))
plt.title('Ingresos por Plan y Mes')
plt.xlabel('Mes')
plt.ylabel('Ingresos ($))')
plt.show()



#  Hipotesis Nula : Los ingresos  de los usuarios de los planes Ultimate y Surf son iguales.
# 
# Hipotesis Alternativa: No hay diferencia significativa en los ingresos totales entre los usuarios del plan "surf" y los usuarios del plan "ultimate"     


# In[82]:


# Prueba las hipótesis
# Los ingresos  de los usuarios de los planes Ultimate y Surf son iguales.
ingresos_surf = tabla_de_consumo_planes[tabla_de_consumo_planes['plan_id'] == 'surf']['ingreso_total']
ingresos_ultimate = tabla_de_consumo_planes[tabla_de_consumo_planes['plan_id'] == 'ultimate']['ingreso_total']
alpha=0.05

#resultados = ttest_ind(ingresos_surf, ingresos_ultimate, equal_var=False)
resultados_t_test = stats.ttest_ind(ingresos_surf, ingresos_ultimate, equal_var=False)

print(f'p-value:{resultados_t_test.pvalue}')
if resultados_t_test.pvalue < alpha:
    print("Podemos rechazar la hipótesis nula")
else:
    print("No podemos rechazar la hipotesis nula")



#     
# Hemos llevado a cabo un análisis estadístico para comparar los ingresos totales entre dos planes de consumo diferentes, "surf" y "ultimate". La finalidad de este análisis es determinar si no existe diferencia en los ingresos generados por los usuarios de estos dos planes.
# Los resultados de nuestro análisis indican que los ingresos generados por los planes "surf" y "ultimate" no son iguales. Esta diferencia es estadísticamente significativa, lo que sugiere que la elección del plan influye en los ingresos totales obtenidos. 


# In[84]:


# Hipotesis Alternativa :
#No hay diferencia significativa en los ingresos totales entre los usuarios del plan "surf" y los usuarios del plan "ultimate"

ingresos_surf = tabla_de_consumo_planes[tabla_de_consumo_planes['plan_id'] == 'surf']['ingreso_total']
ingresos_ultimate = tabla_de_consumo_planes[tabla_de_consumo_planes['plan_id'] == 'ultimate']['ingreso_total']


resultados_levene=stats.levene(ingresos_surf, ingresos_ultimate)
print(f'p-value:{resultados_levene.pvalue}')
if resultados_levene.pvalue < alpha:
    print("Podemos rechazar la hipótesis nula")
else:
    print("No podemos rechazar la hipotesis nula")



# La finalidad de este análisis es determinar si existe una diferencia significativa en los ingresos generados por los usuarios de estos dos planes.
# Dado que el valor p  es menor que nuestro nivel de significancia α (0.05)
# rechazamos la hipótesis nula. Esto significa que hay una diferencia significativa en los ingresos totales entre los usuarios del plan "surf" y los usuarios del plan "ultimate". 


# [Prueba la hipótesis de que el ingreso promedio de los usuarios del área NY-NJ es diferente al de los usuarios de otras regiones.]

# In[75]:


#Prueba la hipótesis de que el ingreso promedio de los usuarios del área NY-NJ es diferente al de los usuarios de otras regiones

# Paso 1: Extraer y agrupar los datos por región
ny_nj_users = user_df[user_df['city'].str.contains('NY|NJ')]
other_users = user_df[~user_df['city'].str.contains('NY|NJ')]

# Ingresos por región
ingresos_ny_nj = tabla_de_consumo_planes[tabla_de_consumo_planes['user_id'].isin(ny_nj_users['user_id'])]['ingreso_total']
ingresos_otros = tabla_de_consumo_planes[tabla_de_consumo_planes['user_id'].isin(other_users['user_id'])]['ingreso_total']

# Paso 3: Prueba t de Student para dos muestras independientes
t_stat, p_value = stats.ttest_ind(ingresos_ny_nj, ingresos_otros, equal_var=False)

print(f"t-statistic={t_stat}, p-value={p_value}")




# In[76]:


# Prueba las hipótesis
# Interpretación de resultados
alpha = 0.05
if p_value < alpha:
    print("Rechazamos la hipótesis nula. El ingreso promedio de los usuarios del área NY-NJ es significativamente diferente al de los usuarios de otras regiones.")
else:
    print("No podemos rechazar la hipótesis nula. No hay evidencia suficiente para afirmar que el ingreso promedio de los usuarios del área NY-NJ es diferente al de los usuarios de otras regiones.")




# Hemos llevado a cabo un análisis estadístico para comparar los ingresos de los usarios del área NY-NJ es diferente a la 
# de otras regiones.
# Dado que el valor p  es mayor que nuestro nivel de significancia α (0.05)
# No rechazamos la hipótesis nula. Esto significa que hay una diferencia  en los ingresos totales entre los usuarios de la zona NY-NJ son diferentes a la de otras regiones.


# Del analisis efectuado en relacion con los ingresos promedios por cada plan , se encontró una diferencia significativa, dado que se percibe mucho más ingresos en el plan surf , esto debido a el exceso de consumo que hacen los usuarios de este plan en comparación con el plan ultimate.
# Existe una clara varaiación en cuanto a los niveles promedio de ingreso generados por cada tipo de plan, lo puede indicar diferencias importantes en términos del valor percibido o del uso realizado por clientes suscritos a cada uno de estos planes. Estas conclusiones podrían ser útiles para tomar decisiones estratégicas relacionadas con la fijación de precios, segmentación del mercado o diseño de nuevos servicios para mejorar la rentabilidad y satisfacción del cliente.       


 # De acuerdo a la muestra de 500 cientes de la Empresa Megaline, se tiene la información de los usuarios, ciudad,
# cantidad de llamadas, mensajes y datos usados asi como también las tarifas por los 2 planes que tiene Surf y Ultimate.
# Por lo que para su mejor ánalisis es necesario la importación de la información y las librerias necesarias para 
# python.
# Se revisaron los datos para identificar valores nulos, duplicados o atípicos.
# En la información de los Dataframe no se encontraron muchos valores ausentes por lo que se opto 
# por reemplazarlos con 0.
# Se realizaron conversiones de tipos de datos necesario en las fechas.
# Al tener la información en distintos Dataframe , se opto por unir la información necesaria de las columnas relevantes
# para el ánalisis ('planes_id','mes') valores en común para tener la información en específico en una Dataframe
# que nos muestre la información por usuario , el plan que le corresponde así como tambíen el uso de datos, mens
# Se crearon nuevas columnas a partir de los datos existentes para proporcionar información adicional relevante para el análisis,
# que contienen la información de la cantidad de servicios adicionales que consumieron los usuarios , par poder determinar los 
# ingresos en total incluyendo el costo base de la tarifa y los costos excedentes.
# En el estudio de comportamiento se decidió hallar estadísticas descriptivas para resumir características clave 
# del conjunto de datos (media,mediana,moda,desviación estandar) asi como también se realizaron las visualizacion con el uso 
# de gráficos (barras,histogramas,boxplot y lineplot) , que nos permiten visualizar el comportamiento y la evolución 
# del consumo de los usuarios y determinar que plan genera un mayor ingreso para la empresa , que como se puede visualizar 
# en el analisis corresponde al plan Surf , que está relacionada a los excedentes de servicios que utilizan los usuarios de 
# este plan.
# Por último establecimos Hipotesis Nula (Los ingresos son iguales en ambos planes) y de acuerdo al ánalisis hecho es 
# y los cálculos determinamos que debemos de rechazar la Hipótesis Nula.
# Por lo que se concluye por el ánalisis hecho el plan que genera mayores ingresos es el plan de Surf.

# 

# 
