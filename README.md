# 📶 Sprint 5 – Análisis Estadístico de Datos (Caso Megaline)

## 📌 Descripción del Proyecto

Este proyecto forma parte del Sprint 5 del programa de Data Science en TripleTen. El objetivo principal es aplicar técnicas de análisis estadístico y pruebas de hipótesis sobre los datos de **Megaline**, una empresa de telecomunicaciones que ofrece planes prepago a sus clientes.

Se realiza un estudio de comportamiento de clientes de Megaline para analizar cuál de las tarifas (`Surf` o `Ultimate`) genera mayores ingresos, y cómo varía el comportamiento de uso por región geográfica.

## 🎯 Propósito del Análisis

- Analizar el comportamiento de los clientes por uso de llamadas, mensajes y navegación por internet.
- Calcular ingresos mensuales por cliente según su tarifa.
- Evaluar si existen diferencias significativas en el ingreso promedio entre planes.
- Probar estadísticamente hipótesis sobre diferencias por plan y por región.

## 🧰 Funcionalidades del proyecto

### Etapa 1: Exploración y limpieza
- Conversión de tipos de datos y tratamiento de valores nulos.
- Revisión de errores en las tablas: `calls`, `messages`, `internet`, `plans`, `users`.

### Etapa 2: Agregación por usuario y mes
- Cálculo de:
  - Minutos usados
  - SMS enviados
  - Megabytes consumidos
  - Ingresos mensuales
- Redondeo correcto por reglas comerciales de Megaline.

### Etapa 3: Análisis estadístico
- Estadísticas descriptivas (media, varianza, desviación estándar).
- Histogramas por plan para llamadas, SMS, datos y facturación.
- Análisis del comportamiento mensual por plan.

### Etapa 4: Pruebas de hipótesis
- Comparación del ingreso mensual promedio:
  - Entre planes `Surf` vs. `Ultimate`
  - Entre región Nueva York/Nueva Jersey vs. otras
- Prueba de hipótesis con nivel de significancia definido.

## 📁 Archivos utilizados

- `megaline_calls.csv`
- `megaline_internet.csv`
- `megaline_messages.csv`
- `megaline_plans.csv`
- `megaline_users.csv`

## 📊 Herramientas utilizadas

- Python  
- pandas  
- matplotlib  
- seaborn  
- scipy.stats

---

📌 Proyecto desarrollado como parte del Sprint 5 en el programa de formación en Ciencia de Datos de **TripleTen**.
