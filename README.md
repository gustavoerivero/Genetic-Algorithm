# 🧬 Laboratorio de IA: Mini-Proyecto de Algoritmos Genéticos

Este proyecto implementa un **Algoritmo Genético Simple (AGS)** para la optimización de funciones matemáticas complejas. Fue desarrollado como parte de la evaluación de la asignatura **Laboratorio de Programación en IA** (Maestría en Ciencias de la Computación).

La solución utiliza una **codificación binaria** y los operadores genéticos clásicos para encontrar el mínimo global de dos funciones de prueba: **Schwefel** (30 dimensiones) y **Six-Hump Camel Back** (2 dimensiones).

## 🚀 Características Principales

* **Interfaz Gráfica Interactiva:** Construida con [Streamlit](https://streamlit.io/) para la configuración dinámica de parámetros (población, probabilidades, generaciones).
* **Motor Evolutivo:** Implementado con la librería **DEAP** en Python.
* **Visualización en Tiempo Real:**
  * Curvas de convergencia (Fitness vs. Generaciones).
  * Visualización 3D interactiva de la topología (para problemas de 2 variables y representación simplificada de Schwefel).
* **Exportación de Datos:** Descarga de resultados históricos en formato CSV.
* **Modularidad:** Código estructurado en módulos (lógica, configuración, gráficos).

## 📂 Estructura del Proyecto

```text
MiniProyecto_IA/
│
├── app.py                  # Punto de entrada de la aplicación (GUI Streamlit)
├── README.md               # Documentación del proyecto
├── requirements.txt        # Dependencias necesarias
│
├── config/
│   └── config.py           # Configuración central y parámetros por defecto
│
└── modules/
    ├── functions.py        # Funciones objetivo (Schwefel, Camel Back) y decodificación binaria
    ├── ga.py               # Configuración del motor DEAP (Toolbox)
    └── graphics.py         # Generación de gráficas (Matplotlib/Plotly)
```

## ⚙️ Fundamentos Teóricos e Implementación

Para cumplir con los requerimientos académicos, se implementaron los siguientes operadores específicos:

1. **Representación (Codificación):** Binaria. El espacio de búsqueda real continuo se mapea a una cadena de bits utilizando una ecuación de decodificación lineal.
2. **Selección: Rueda de Ruleta** (Roulette Wheel)**.** Se aplica una transformación de fitness para convertir el problema de minimización en uno de maximización apto para la ruleta.
3. **Cruce: Un punto** (One-Point Crossover).
4. **Mutación: Simple** (Bit-Flip Mutation), con probabilidad independiente por bit.

### Funciones de Prueba

| **Función**            | **Dimensiones (n)** | **Rango**           | **Óptimo Global**        | **Características**                     |
| ----------------------------- | ------------------------- | ------------------------- | ------------------------------- | ---------------------------------------------- |
| **Six-Hump Camel Back** | 2                         | **$[-5, 5]$**     | **$\approx -1.0316$**   | Multimodal, suave.                             |
| **Schwefel**            | 30                        | **$[-500, 500]$** | **$\approx -12569.48$** | Altamente multimodal, muchos mínimos locales. |

## 🛠️ Instalación y Requisitos

Asegúrate de tener **Python 3.8+** instalado.

1. **Clonar o descargar el repositorio.**
2. **Instalar dependencias:** Se recomienda usar un entorno virtual. Ejecuta el siguiente comando para instalar las librerías necesarias:

**Bash**

```
pip install streamlit deap numpy matplotlib pandas plotly openpyxl
```

## ▶️ Ejecución

Para iniciar la interfaz gráfica, ejecuta el siguiente comando desde la raíz del proyecto:

**Bash**

```
streamlit run app.py
```

Esto abrirá automáticamente una pestaña en tu navegador web donde podrás interactuar con el algoritmo.

## 📊 Uso de la Aplicación

1. **Selección del Problema:** Elige entre "Schwefel" o "Camel Back" en el panel lateral. Los parámetros recomendados se cargarán automáticamente.
2. **Configuración:** Ajusta el tamaño de la población, generaciones y probabilidades de cruce/mutación si deseas experimentar.
3. **Ejecución:** Haz clic en  **"Ejecutar algoritmo"** .
4. **Análisis:** Observa la gráfica de convergencia, la visualización 3D y descarga el CSV con los datos de la evolución si lo necesitas para tu informe.

---

**Autores:** Gustavo Rivero, Zaidibeth Ramos
**Asignatura:** Laboratorio de Programación en IA
**Fecha:** Diciembre 2025
"# Genetic-Algorithm" 
