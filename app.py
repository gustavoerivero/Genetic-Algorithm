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

if 'selected_problem_index' not in st.session_state:
    st.session_state.selected_problem_index = 1


def update_problem():
    problem_name = st.session_state.problem_selector
    if "Schwefel" in problem_name:
        config.PROBLEM_ID = 1
    else:
        config.PROBLEM_ID = 2


@st.cache_data(show_spinner=False)
def run_genetic_algorithm(prob_id, pop_size, gens, cx_prob, mut_prob):

    config.PROBLEM_ID = prob_id
    config.PROBLEMS[prob_id]["population_size"] = pop_size
    config.PROBLEMS[prob_id]["generations"] = gens

    random.seed(42)
    toolbox = ga.setup_ga()
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=cx_prob, mutpb=mut_prob,
        ngen=gens, stats=stats, halloffame=hof, verbose=False
    )

    best_ind = hof[0]
    return logbook, best_ind


st.sidebar.header("🛠️ Configuración")

problem_name = st.sidebar.selectbox(
    "Función Objetivo",
    ["Schwefel (30 vars)", "Six-Hump Camel Back (2 vars)"],
    index=1,
    key="problem_selector",
    on_change=update_problem
)


if "Schwefel" in problem_name:
    selected_id = 1
else:
    selected_id = 2

current_params = config.get_problem_config(config.PROBLEM_ID)

st.sidebar.subheader("Parámetros AG")
pop_size = st.sidebar.number_input(
    "Población", 
    10, 
    5000, 
    current_params["population_size"], 
    10, 
    key=f"p_{config.PROBLEM_ID}"
)

generations = st.sidebar.number_input(
    "Generaciones", 
    10, 
    1000, 
    current_params["generations"], 
    10, 
    key=f"g_{config.PROBLEM_ID}"
)

crossover_prob = st.sidebar.slider(
    "Prob. Cruce", 
    0.0, 
    1.0, 
    config.CROSSOVER_PROB, 
    key=f"c_{config.PROBLEM_ID}"
)

mutation_prob = st.sidebar.slider(
    "Prob. Mutación", 
    0.0, 
    0.1, 
    config.MUTATION_PROB, 
    format="%.3f", 
    step=0.001,
    key=f"m_{config.PROBLEM_ID}"
)

if st.button("🚀 Ejecutar Algoritmo Genético", type="primary"):
    with st.spinner("Ejecutando algoritmo genético..."):
        try:
            logbook, best_ind = run_genetic_algorithm(
                config.PROBLEM_ID, pop_size, generations, crossover_prob, mutation_prob
            )

            real_val = current_params["offset_roulette"] - best_ind.fitness.values[0]
            decoded_vars = functions.decode_chromosome(
                best_ind,
                current_params["var_range"][0],
                current_params["var_range"][1],
                current_params["n_vars"]
            )

            found_gen = 0

            target_fitness = best_ind.fitness.values[0]

            for record in logbook:
                if record['max'] >= target_fitness:
                    found_gen = record['gen']
                    break

            st.divider()

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Mínimo encontrado", f"{real_val:.5f}")
            c2.metric("Generaciones", f"{generations}")
            c3.metric("Encontrado en Gen.", f"{found_gen}")
            c4.metric("Población", f"{pop_size}")

            st.subheader("📉 Convergencia")
            fig_conv = graphics.plot_convergence(logbook)
            st.pyplot(fig_conv)

            if config.PROBLEM_ID == 2:
                st.subheader("🏔️ Visualización 3D")
                fig_3d = graphics.plot_3d_surface(decoded_vars)
                st.plotly_chart(fig_3d, use_container_width=True)
            elif config.PROBLEM_ID == 1:
                st.subheader("🏔️ Topología Referencial (2D)")
                fig_3d = graphics.plot_schwefel_surface_2d()
                st.plotly_chart(fig_3d, use_container_width=True)

            st.subheader("🧬 Variables (Genotipo)")
            st.code(str(decoded_vars), language="python")

            df_log = pd.DataFrame(logbook)
            df_log['valor_real'] = current_params["offset_roulette"] - df_log['max']
            st.download_button(
                "Descargar CSV",
                df_log.to_csv(index=False).encode('utf-8'),
                f'resultados_{current_params["name"]}.csv',
                "text/csv"
            )

        except Exception as e:
            st.error(f"Ocurrió un error (Posible timeout por recursos): {e}")