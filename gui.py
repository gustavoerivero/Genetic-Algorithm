import tkinter as tk
import random
from tkinter import ttk, messagebox
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import config.config as config
import modules.ga as ga
import modules.functions as functions
from deap import tools, algorithms

class GeneticApp:
    random.seed(42)

    def __init__(self, root):
        self.root = root
        self.root.title("Laboratorio de IA - Algoritmos Genéticos")
        self.root.geometry("1150x800")

        self.problem_var = tk.StringVar(value="Six-Hump Camel Back")
        self.pop_var = tk.IntVar(value=200)
        self.gen_var = tk.IntVar(value=100)
        self.cx_prob_var = tk.DoubleVar(value=0.8)
        self.mut_prob_var = tk.DoubleVar(value=0.01)
        
        self.res_min = tk.StringVar(value="---")
        self.res_gen = tk.StringVar(value="---")
        self.status_msg = tk.StringVar(value="Listo para ejecutar")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(main_frame, relief="groove", borderwidth=2, padding="5")
        header_frame.pack(fill=tk.X, pady=(0, 15))

        lbl_uni = ttk.Label(
            header_frame, 
            text='Universidad Centroccidental "Lisandro Alvarado"', 
            font=("Helvetica", 16, "bold"),
            foreground="#003366"
        )
        lbl_uni.pack(anchor=tk.CENTER, pady=(5, 0))

        lbl_authors = ttk.Label(
            header_frame, 
            text="Zaidibeth Ramos y Gustavo Rivero", 
            font=("Helvetica", 12, "bold"),
            foreground="#444"
        )
        lbl_authors.pack(anchor=tk.CENTER, pady=(5, 5))

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.LabelFrame(content_frame, text="Configuración Experimental", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(left_panel, text="Función Objetivo:").pack(anchor=tk.W)
        problem_cb = ttk.Combobox(left_panel, textvariable=self.problem_var, state="readonly", width=30)
        problem_cb['values'] = ("Six-Hump Camel Back", "Schwefel (30 vars)")
        problem_cb.pack(fill=tk.X, pady=5)
        problem_cb.bind("<<ComboboxSelected>>", self.update_defaults)

        labels = ["Población:", "Generaciones:", "Prob. Cruce:", "Prob. Mutación:"]
        vars_list = [self.pop_var, self.gen_var, self.cx_prob_var, self.mut_prob_var]
        
        for lbl, var in zip(labels, vars_list):
            ttk.Label(left_panel, text=lbl).pack(anchor=tk.W, pady=(10, 0))
            ttk.Entry(left_panel, textvariable=var).pack(fill=tk.X)

        self.btn_run = ttk.Button(left_panel, text="🚀 EJECUTAR ALGORITMO", command=self.start_thread)
        self.btn_run.pack(fill=tk.X, pady=20)

        ttk.Label(left_panel, textvariable=self.status_msg, foreground="blue", wraplength=200).pack(fill=tk.X)

        res_frame = ttk.LabelFrame(left_panel, text="Métricas de Desempeño", padding="10")
        res_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(res_frame, text="Mínimo Global Encontrado:").pack(anchor=tk.W)
        ttk.Label(res_frame, textvariable=self.res_min, font=("Arial", 14, "bold"), foreground="green").pack(anchor=tk.W)
        
        ttk.Label(res_frame, text="Detectado en Generación:").pack(anchor=tk.W, pady=(5,0))
        ttk.Label(res_frame, textvariable=self.res_gen, font=("Arial", 11, "bold")).pack(anchor=tk.W)

        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_conv = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_conv, text="📉 Curva de Convergencia")
        
        self.fig_conv, self.ax_conv = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas_conv = FigureCanvasTkAgg(self.fig_conv, master=self.tab_conv)
        self.canvas_conv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.tab_3d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_3d, text="🏔️ Topología 3D del Terreno")
        
        self.fig_3d = plt.figure(figsize=(5, 4), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_defaults(self, event=None):
        selection = self.problem_var.get()
        if "Schwefel" in selection:
            self.pop_var.set(2000)
            self.gen_var.set(500)
            self.mut_prob_var.set(0.002)
        else:
            self.pop_var.set(200)
            self.gen_var.set(100)
            self.mut_prob_var.set(0.01)

    def start_thread(self):
        self.btn_run.config(state=tk.DISABLED)
        self.status_msg.set("Procesando... Esto puede tardar unos segundos.")
        thread = threading.Thread(target=self.run_algorithm)
        thread.daemon = True
        thread.start()

    def run_algorithm(self):
        try:
            selection = self.problem_var.get()
            prob_id = 1 if "Schwefel" in selection else 2
            
            pop_size = self.pop_var.get()
            gens = self.gen_var.get()
            cx_pb = self.cx_prob_var.get()
            mut_pb = self.mut_prob_var.get()

            config.PROBLEM_ID = prob_id
            config.PROBLEMS[prob_id]["population_size"] = pop_size
            config.PROBLEMS[prob_id]["generations"] = gens
            config.CROSSOVER_PROB = cx_pb
            config.MUTATION_PROB = mut_pb

            params = config.get_problem_config(prob_id)
            toolbox = ga.setup_ga()
            pop = toolbox.population(n=pop_size)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("max", np.max)

            pop, logbook = algorithms.eaSimple(
                pop, toolbox, 
                cxpb=cx_pb, mutpb=mut_pb, 
                ngen=gens, stats=stats, halloffame=hof, verbose=False
            )

            best_ind = hof[0]
            real_val = params["offset_roulette"] - best_ind.fitness.values[0]
            
            target = best_ind.fitness.values[0]
            found_gen = 0
            for record in logbook:
                if record['max'] >= target:
                    found_gen = record['gen']
                    break
            
            decoded_vars = functions.decode_chromosome(
                best_ind, params["var_range"][0], params["var_range"][1], params["n_vars"]
            )

            gen_log = logbook.select("gen")
            fit_max = logbook.select("max")
            real_values_log = [params["offset_roulette"] - f for f in fit_max]

            self.root.after(0, self.update_gui_results, real_val, found_gen, gen_log, real_values_log, prob_id, decoded_vars)

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error de Ejecución", error_msg))
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_msg.set("Error. Verifique parámetros."))

    def update_gui_results(self, val, gen, x_data, y_data, prob_id, best_vars):
        self.res_min.set(f"{val:.5f}")
        self.res_gen.set(str(gen))
        self.status_msg.set("✅ Optimización completada exitosamente.")
        self.btn_run.config(state=tk.NORMAL)

        self.ax_conv.clear()
        self.ax_conv.plot(x_data, y_data, color="#1f77b4", linewidth=2, label="Mejor Fitness")
        self.ax_conv.set_title("Evolución del Fitness por Generación")
        self.ax_conv.set_xlabel("Generaciones")
        self.ax_conv.set_ylabel("Valor de la Función (Minimizar)")
        self.ax_conv.grid(True, linestyle='--', alpha=0.6)
        self.ax_conv.legend()
        self.canvas_conv.draw()

        self.ax_3d.clear()
        
        if prob_id == 2:
            x = np.linspace(-2, 2, 60)
            y = np.linspace(-1, 1, 60)
            X, Y = np.meshgrid(x, y)
            Z = (4*X**2 - 2.1*X**4 + (1/3)*X**6) + (X*Y) + (-4*Y**2 + 4*Y**4)
            
            self.ax_3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, linewidth=0)
            
            bx, by = best_vars[0], best_vars[1]
            bz = (4*bx**2 - 2.1*bx**4 + (1/3)*bx**6) + (bx*by) + (-4*by**2 + 4*by**4)
            self.ax_3d.scatter(bx, by, bz, c='red', s=150, marker='*', label='Mínimo Encontrado', depthshade=False)
            self.ax_3d.set_title("Topología: Six-Hump Camel Back")
            self.ax_3d.legend()

        elif prob_id == 1:
            x = np.linspace(-500, 500, 50)
            y = np.linspace(-500, 500, 50)
            X, Y = np.meshgrid(x, y)
            Z = (-X * np.sin(np.sqrt(np.abs(X)))) + (-Y * np.sin(np.sqrt(np.abs(Y))))
            
            self.ax_3d.plot_surface(X, Y, Z, cmap=cm.jet, alpha=0.7)
            self.ax_3d.set_title("Schwefel (Representación 2D)")
            self.status_msg.set("Nota: Schwefel se muestra en 2D solo como referencia topológica.")

        self.canvas_3d.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticApp(root)
    root.mainloop()