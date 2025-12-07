import streamlit as st
import numpy as np
import random
import pandas as pd
from deap import tools, algorithms

import config.config as config
import modules.ga as ga
import modules.graphics as graphics
import modules.functions as functions


st.set_page_config(page_title="Optimización Genética", layout="wide")
st.title("🧬 Algoritmos Genéticos")
st.markdown("Ejecución y Visualización de Algoritmos Genéticos")

st.sidebar.header("🛠️ Configuración")

problem_name = st.sidebar.selectbox(
    "Función Objetivo",
    ["Schwefel (30 vars)", "Six-Hump Camel Back (2 vars)"],
    index=1
)


if "Schwefel" in problem_name:
    selected_id = 1
else:
    selected_id = 2

config.PROBLEM_ID = selected_id
defaults = config.get_problem_config(selected_id)

st.sidebar.subheader("Parámetros AG")

pop_size = st.sidebar.number_input(
    "Población", 
    min_value=10, 
    max_value=5000, 
    value=defaults["population_size"],
    step=10,
    key=f"pop_{selected_id}"
)

generations = st.sidebar.number_input(
    "Generaciones", 
    min_value=10, 
    max_value=1000, 
    value=defaults["generations"],
    step=10,
    key=f"gen_{selected_id}"
)

crossover_prob = st.sidebar.slider(
    "Prob. Cruce", 
    0.0, 1.0, 
    config.CROSSOVER_PROB,
    key=f"cross_{selected_id}"
)

mutation_prob = st.sidebar.slider(
    "Prob. Mutación", 
    0.0, 0.1, 
    config.MUTATION_PROB, 
    format="%.3f",
    step=0.001,
    key=f"mut_{selected_id}"
)

config.PROBLEMS[config.PROBLEM_ID]["population_size"] = pop_size
config.PROBLEMS[config.PROBLEM_ID]["generations"] = generations
config.CROSSOVER_PROB = crossover_prob
config.MUTATION_PROB = mutation_prob

if st.button("🚀 Ejecutar algoritmo", type="primary"):
    random.seed(42)
    
    
    progress_bar = st.progress(0, text="Inicializando población...")
    
    
    with st.spinner('Procesando generaciones...'):
        toolbox = ga.setup_ga()
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        
        
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, 
            cxpb=crossover_prob, mutpb=mutation_prob, 
            ngen=generations, stats=stats, halloffame=hof, verbose=False
        )
        
        progress_bar.progress(100, text="¡Optimización completada!")


    best_ind = hof[0]
    real_val = defaults["offset_roulette"] - best_ind.fitness.values[0]
    decoded_vars = functions.decode_chromosome(
        best_ind, 
        defaults["var_range"][0], 
        defaults["var_range"][1], 
        defaults["n_vars"]
    )


    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Mínimo Encontrado", f"{real_val:.5f}")
    col2.metric("Generaciones", generations)
    col3.metric("Población final", pop_size)
    
    st.success(f"Mejor solución encontrada para **{defaults['name']}**")


    st.subheader("📉 Convergencia del Fitness")
    fig_conv = graphics.plot_convergence(logbook)
    st.pyplot(fig_conv)


    if config.PROBLEM_ID == 1:
        st.info("⚠️ Nota: El problema se resuelve en 30 dimensiones. Esta gráfica es una representación visual del terreno en solo 2 dimensiones para observar sus múltiples mínimos locales.")
        fig_3d = graphics.plot_schwefel_surface_2d()
        st.plotly_chart(fig_3d, use_container_width=True)

    elif config.PROBLEM_ID == 2:
        st.subheader("🏔️ Visualización 3D del terreno")
        st.markdown("El punto rojo indica el mínimo encontrado por el algoritmo.")
        fig_3d = graphics.plot_3d_surface(decoded_vars)
        st.plotly_chart(fig_3d, use_container_width=True)


    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("🧬 Variables de decisión (Genotipo)")
        st.code(str(decoded_vars), language="python")
        
    with c2:
        st.subheader("💾 Exportar")
        
        df_log = pd.DataFrame(logbook)
        df_log['valor_real'] = defaults["offset_roulette"] - df_log['max']
        
        st.download_button(
            label="Descargar CSV",
            data=df_log.to_csv(index=False).encode('utf-8'),
            file_name=f'resultados_{defaults["name"]}.csv',
            mime='text/csv',
            use_container_width=True
        )