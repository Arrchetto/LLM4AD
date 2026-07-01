# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/12/05
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

import os
import shutil

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

GUI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(GUI_DIR)
METHOD_DIR = os.path.join(REPO_ROOT, 'llm4ad', 'method')
TASK_DIR = os.path.join(REPO_ROOT, 'llm4ad', 'task')

sys.path.append(REPO_ROOT)

import time
import csv
import math
from datetime import datetime
import pytz
import tkinter as tk
from tkinter import ttk as tkttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import json
import webbrowser
import multiprocessing
from llm4ad.gui import main_gui
import threading
import ttkbootstrap as ttk
import subprocess
import yaml

##########################################################


def _asset_path(*parts):
    return os.path.join(GUI_DIR, *parts)


selected_algo = None
selected_problem = None

process1 = None
thread1 = None

stop_thread = False
have_stop_thread = False
monitor_sample_count = 0
monitor_max_sample_nums = 0
monitor_finalized = True

method_para_entry_list = []
method_para_value_type_list = []
method_para_value_name_list = []

problem_listbox = None
default_problem_index = None
objectives_var = None
problem_para_entry_list = []
problem_para_value_type_list = []
problem_para_value_name_list = []

llm_para_entry_list = []
llm_para_value_name_list = ['name', 'host', 'key', 'model']
llm_para_default_value_list = ['HttpsApi', 'yunwu.ai', '', 'gpt-4o-mini']
llm_para_placeholder_list = ['HttpsApi', 'yunwu.ai', 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', 'gpt-4o-mini']
formal_save_var = None

default_method = 'eoh'
default_problem = ['admissible_set', 'car_mountain', 'bactgrow']

log_dir = None
figures = None
ax = None
canvas = None

##########################################################

class PlaceholderEntry(ttk.Entry):
    def __init__(self, master=None, placeholder="Enter text here", color='grey', bootstyle='default', width=30):
        super().__init__(master, bootstyle=bootstyle, width=width)

        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['foreground']

        self.bind("<FocusIn>", self._clear_placeholder)
        self.bind("<FocusOut>", self._add_placeholder)

        self._add_placeholder(force=True)

        self.have_content = False

    def _add_placeholder(self, event=None, force=False):
        self.have_content = True
        if not self.get() or force:
            self.configure(foreground=self.placeholder_color)
            self.delete(0, 'end')
            self.insert(0, self.placeholder)
            self.have_content = False

    def _clear_placeholder(self, event=None):
        if self.get() == self.placeholder and str(self['foreground']) == str(self.placeholder_color):
            self.delete(0, "end")
            self.configure(foreground=self.default_fg_color)

class ScrollableFrame(ttk.Frame):
    """Frame with scrollbar"""

    def __init__(self, container, max_height=250, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # Create Canvas and Scrollbar
        canvas = tk.Canvas(self, bg='white', highlightthickness=0, height=max_height)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview, bootstyle="round")
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel
        self.scrollable_frame.bind('<Enter>', lambda e: self._bind_mousewheel(canvas))
        self.scrollable_frame.bind('<Leave>', lambda e: self._unbind_mousewheel(canvas))

    def _bind_mousewheel(self, canvas):
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def _unbind_mousewheel(self, canvas):
        canvas.unbind_all("<MouseWheel>")


##########################################################

def draw_horizontal_line(parent_frame, width=150):
    line_canvas = tk.Canvas(parent_frame, width=width, height=25, bg='white', highlightthickness=0)
    line_canvas.pack(pady=0)
    line_canvas.create_line(0, 15, width, 15, fill='black')


def open_doc_link():
    webbrowser.open_new("https://llm4ad-doc.readthedocs.io/en/latest/")


def open_github_link():
    webbrowser.open_new("https://github.com/Optima-CityU/LLM4AD")


def open_website_link():
    webbrowser.open_new("http://www.llm4ad.com/index.html")


def open_qq_link():
    webbrowser.open_new("https://qm.qq.com/cgi-bin/qm/qr?k=4Imf8bn_d99-QXVcEJfOwCSD1KkcpbcD&jump_from=webapi&authKey=JtSmFh8BNKM97+TGnUdDgvT69TDTbo4UaLwgrZJSlsYqmVoCca/a5awU+TXt4zYB")


def open_folder():
    global log_dir

    if os.path.exists(log_dir):
        if os.name == 'nt':  # Windows
            os.startfile(log_dir)
        elif os.name == 'posix':  # Unix-like
            subprocess.run(['open', log_dir])


##########################################################

def on_algo_select(event):
    global selected_algo
    if algo_listbox.curselection():
        selected_algo = algo_listbox.get(algo_listbox.curselection())
        show_algorithm_parameters(selected_algo)


def on_problem_select(event):
    global selected_problem
    if problem_listbox.curselection():
        selected_problem = problem_listbox.get(problem_listbox.curselection())
        show_problem_parameters(selected_problem)


def show_algorithm_parameters(algo_name):
    global method_para_entry_list
    global method_para_value_type_list
    global method_para_value_name_list
    clear_algo_param_frame()

    algo_param_frame['text'] = f"{algo_name}"

    required_parameters, value_type, default_value = get_required_parameters(
        path=os.path.join(METHOD_DIR, algo_name, 'paras.yaml'))
    method_para_value_name_list = required_parameters
    method_para_value_type_list = value_type

    # Create scrollable area within algo_param_frame
    scroll_frame = ScrollableFrame(algo_param_frame, max_height=200)
    scroll_frame.pack(fill='both', expand=True)
    inner_frame = scroll_frame.scrollable_frame

    for i in range(len(required_parameters)):
        if i != 0:
            ttk.Label(inner_frame, text=required_parameters[i] + ':').grid(row=i - 1, column=0, sticky='w', padx=5,
                                                                           pady=5)
        method_para_entry_list.append(ttk.Entry(inner_frame, width=10, bootstyle="primary"))
        if i != 0:
            method_para_entry_list[-1].grid(row=i - 1, column=1, sticky='ew', padx=5, pady=5)
        if default_value[i] is not None:
            method_para_entry_list[-1].insert(0, str(default_value[i]))

    inner_frame.grid_columnconfigure(0, weight=1)
    inner_frame.grid_columnconfigure(1, weight=2)


def show_problem_parameters(problem_name):
    global problem_para_entry_list
    global problem_para_value_type_list
    global problem_para_value_name_list
    clear_problem_param_frame()

    problem_param_frame['text'] = f"{problem_name}"

    if problem_name[-8:] == 'co_bench':
        yaml_file_path = os.path.join(TASK_DIR, objectives_var.get(), 'co_bench', problem_name, 'paras.yaml')
    else:
        yaml_file_path = os.path.join(TASK_DIR, objectives_var.get(), problem_name, 'paras.yaml')

    required_parameters, value_type, default_value = get_required_parameters(path=yaml_file_path)
    problem_para_value_type_list = value_type
    problem_para_value_name_list = required_parameters
    for i in range(len(required_parameters)):
        if i != 0:
            ttk.Label(problem_param_frame, text=required_parameters[i] + ':').grid(row=i - 1, column=0, sticky='nsew', padx=5, pady=10)
        problem_para_entry_list.append(ttk.Entry(problem_param_frame, width=10, bootstyle="warning"))
        if i != 0:
            problem_para_entry_list[-1].grid(row=i - 1, column=1, sticky='nsew', padx=5, pady=10)
            problem_param_frame.grid_rowconfigure(i - 1, weight=1)
        if default_value[i] is not None:
            problem_para_entry_list[-1].insert(0, str(default_value[i]))
    problem_param_frame.grid_columnconfigure(0, weight=1)
    problem_param_frame.grid_columnconfigure(1, weight=2)

    if len(required_parameters) < 5:
        for i in range(len(required_parameters), 5):
            problem_param_frame.grid_rowconfigure(i - 1, weight=1)


def get_required_parameters(path):
    required_parameters = []
    value_type = []
    default_value = []

    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)  # 使用 safe_load 读取 YAML 文件

    for key, value in data.items():
        required_parameters.append(key)
        value_type.append(str(type(value)))
        if value is None:
            default_value.append(value)
        else:
            default_value.append(str(value))

    return required_parameters, value_type, default_value


def _convert_parameter_value(raw_value, value_type):
    """Convert a GUI entry string using the type inferred from YAML."""
    if value_type == "<class 'NoneType'>":
        return raw_value or None
    if value_type == "<class 'int'>":
        return int(raw_value)
    if value_type == "<class 'float'>":
        return float(raw_value)
    if value_type == "<class 'bool'>":
        normalized = raw_value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise ValueError(
            f"Invalid boolean value {raw_value!r}; use true or false."
        )
    return raw_value


def _complete_method_parameters(method_parameters):
    """Preserve explicit sampler concurrency and support legacy GUI configs."""
    completed = dict(method_parameters)
    if "num_samplers" not in completed and "num_evaluators" in completed:
        completed["num_samplers"] = completed["num_evaluators"]
    return completed


def _profiler_name_for_method(method_name):
    """Use method-specific profiler features where they are required."""
    if method_name == "EoH":
        return "EoHProfiler"
    return "ProfilerBase"


def _redact_sensitive_mapping(parameters):
    """Return a printable copy without credential values."""
    sensitive_names = ("key", "secret", "password", "token")
    return {
        key: (
            "<redacted>"
            if any(name in str(key).strip("_").lower() for name in sensitive_names)
            else value
        )
        for key, value in parameters.items()
    }


def _build_method_log_folder(
        method_name,
        problem_name,
        process_start_time,
        formal_save=True):
    """Build the profiler directory under a lowercase method folder."""
    method_folder = str(method_name).strip().lower()
    run_folder = (
        process_start_time.strftime("%Y%m%d_%H%M%S")
        + f"_{problem_name}_{method_name}"
    )
    path_parts = [GUI_DIR, "logs", method_folder]
    if not formal_save:
        path_parts.append("test")
    path_parts.append(run_folder)
    return os.path.join(*path_parts)


def clear_algo_param_frame():
    global method_para_entry_list
    global method_para_value_type_list
    global method_para_value_name_list
    method_para_value_type_list = []
    method_para_value_name_list = []
    method_para_entry_list = []
    for widget in algo_param_frame.winfo_children():
        widget.destroy()


def clear_problem_param_frame():
    global problem_para_entry_list
    global problem_para_value_type_list
    global problem_para_value_name_list
    problem_para_value_type_list = []
    problem_para_value_name_list = []
    problem_para_entry_list = []
    for widget in problem_param_frame.winfo_children():
        widget.destroy()


def discover_task_directories(path):
    """Return runnable task folders, excluding support-only packages."""
    path = os.fspath(path)
    return sorted(
        name
        for name in os.listdir(path)
        if name not in {'__pycache__', '_data', 'co_bench'}
        and os.path.isdir(os.path.join(path, name))
        and os.path.isfile(os.path.join(path, name, 'paras.yaml'))
    )


def problem_type_select(event=None):
    global problem_listbox
    global default_problem_index
    global objectives_var

    default_problem_index = None
    if problem_listbox is not None:
        problem_listbox.destroy()

    problem_listbox = tk.Listbox(problem_frame, height=6, bg='white', selectbackground='lightgray', font=('Comic Sans MS', 12))
    problem_listbox.pack(anchor=tk.NW, fill='both', expand=True, padx=5, pady=5)
    path = os.path.join(TASK_DIR, objectives_var.get())
    for name in discover_task_directories(path):
        problem_listbox.insert(tk.END, name)
        if name in default_problem:
            default_problem_index = problem_listbox.size() - 1

    if objectives_var.get() == 'optimization':
        path = os.path.join(TASK_DIR, objectives_var.get(), 'co_bench')  # todo
        for name in discover_task_directories(path):
            problem_listbox.insert(tk.END, name)

    problem_listbox.bind("<<ListboxSelect>>", on_problem_select)
    on_problem_select(problem_listbox.select_set(default_problem_index))


###############################################################################

def on_plot_button_click():
    global process1
    global thread1
    global log_dir

    try:

        if not check_para():
            tk.messagebox.showinfo("Warning", "Please configure the settings of LLM.")
            return

        llm_para, method_para, problem_para, profiler_para = return_para()

        init_fig(method_para['max_sample_nums'])

        process1 = multiprocessing.Process(target=main_gui, args=(llm_para, method_para, problem_para, profiler_para))
        process1.start()

        log_dir = profiler_para['log_dir']
        thread1 = None
        start_result_monitor(log_dir, method_para['max_sample_nums'])

        plot_button['state'] = tk.DISABLED
        stop_button['state'] = tk.NORMAL
        # doc_button['state'] = tk.DISABLED
        doc_button['state'] = tk.NORMAL

    except ValueError:
        print("Invalid input. Please enter a number.")


def check_para():
    for i in llm_para_entry_list[1:]:
        if not i.have_content:
            return False
    return True


def return_para():
    llm_para = {}
    method_para = {}
    problem_para = {}
    profiler_para = {}

    ####################

    for i in range(len(llm_para_entry_list)):
        llm_para[llm_para_value_name_list[i]] = llm_para_entry_list[i].get()

    for i in range(len(method_para_entry_list)):
        method_para[method_para_value_name_list[i]] = _convert_parameter_value(
            method_para_entry_list[i].get(),
            method_para_value_type_list[i],
        )

    method_para = _complete_method_parameters(method_para)

    for i in range(len(problem_para_entry_list)):
        problem_para[problem_para_value_name_list[i]] = _convert_parameter_value(
            problem_para_entry_list[i].get(),
            problem_para_value_type_list[i],
        )

    ####################

    profiler_para['name'] = _profiler_name_for_method(method_para['name'])

    temp_str1 = problem_para['name']
    temp_str2 = method_para['name']
    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    log_folder = _build_method_log_folder(
        temp_str2,
        temp_str1,
        process_start_time,
        formal_save=formal_save_var.get() if formal_save_var is not None else True,
    )
    profiler_para['log_dir'] = log_folder

    ####################

    print(_redact_sensitive_mapping(llm_para))
    print(method_para)
    print(problem_para)
    print(profiler_para)

    return llm_para, method_para, problem_para, profiler_para

######################################################################

def init_fig(max_sample_nums):
    global stop_thread
    global have_stop_thread
    global thread1
    global process1
    global ax
    global figures
    global canvas

    stop_run()
    value_label.config(text=f"{0} samples")

    stop_thread = False
    have_stop_thread = False

    right_frame_label['text'] = 'Running'

    code_display.config(state='normal')
    code_display.delete(1.0, 'end')
    code_display.config(state='disabled')

    objective_label['text'] = 'Current best objective:'

    for widget in plot_frame.winfo_children():
        widget.destroy()

    font = {
        'family': 'Times New Roman',
        'size': 16
    }

    figures = plt.Figure(figsize=(4, 3), dpi=100)
    ax = figures.add_subplot(111)

    figures.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title(f"Result Display", fontdict=font)

    ax.plot()
    ax.set_xlim(left=0)
    ax.set_xlabel('Samples', fontdict=font)
    ax.set_ylabel('Current best objective', fontdict=font)
    ax.grid(True)

    if max_sample_nums <= 20:
        ax.set_xticks(np.arange(0, max_sample_nums + 1, 1))
    else:
        ticks = np.linspace(0, max_sample_nums, 11)
        ticks = np.round(ticks).astype(int)
        ax.set_xticks(ticks)

    canvas = FigureCanvasTkAgg(figures, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def get_available_sample_count(run_log_dir):
    """Return the number of profiler samples currently persisted on disk."""
    samples_dir = os.path.join(run_log_dir, "samples")
    if not os.path.isdir(samples_dir):
        return 0

    count = 0
    for filename in os.listdir(samples_dir):
        if (
            not filename.startswith("samples_")
            or not filename.endswith(".json")
            or filename == "samples_best.json"
        ):
            continue
        try:
            with open(
                os.path.join(samples_dir, filename),
                encoding="utf-8",
            ) as file:
                count += len(json.load(file))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return count


def schedule_result_poll(delay_ms=500):
    """Schedule result polling on Tkinter's main event loop."""
    app_root = globals().get("root")
    if app_root is not None:
        app_root.after(delay_ms, poll_results)


def start_result_monitor(run_log_dir, max_sample_nums):
    """Start a main-thread result monitor for the current experiment."""
    global log_dir
    global monitor_sample_count
    global monitor_max_sample_nums
    global monitor_finalized
    global have_stop_thread

    log_dir = run_log_dir
    monitor_sample_count = 0
    monitor_max_sample_nums = max_sample_nums
    monitor_finalized = False
    have_stop_thread = False
    schedule_result_poll(delay_ms=0)


def _refresh_results_to_latest():
    """Refresh the live GUI once using every sample currently on disk."""
    global monitor_sample_count

    available_count = get_available_sample_count(log_dir)
    if available_count <= monitor_sample_count:
        return

    _, algorithm, best_objective = plot_fig(
        available_count,
        log_dir,
        monitor_max_sample_nums,
    )
    display_plot(available_count - 1)
    if algorithm is not None:
        display_alg(algorithm)
    objective_label["text"] = f"Current best objective:{best_objective}"
    monitor_sample_count = available_count


def finalize_result_monitor():
    """Perform the final refresh and export exactly once."""
    global monitor_finalized
    global have_stop_thread

    if monitor_finalized:
        return

    try:
        _refresh_results_to_latest()
    except Exception as error:
        print(f"Failed to perform final GUI refresh: {error}")

    try:
        export_convergence_artifacts(log_dir)
    except Exception as error:
        print(f"Failed to export convergence artifacts: {error}")

    if except_error():
        tk.messagebox.showerror("Error", "Except Error. Please check the terminal.")
        right_frame_label["text"] = "Error"
    elif stop_thread:
        right_frame_label["text"] = "Stopped"
    else:
        right_frame_label["text"] = "Finished"

    monitor_finalized = True
    have_stop_thread = True
    plot_button["state"] = tk.NORMAL
    stop_button["state"] = tk.DISABLED


def poll_results():
    """Poll persisted results and update Tk widgets only on the main thread."""
    if monitor_finalized:
        return

    try:
        _refresh_results_to_latest()
    except Exception as error:
        print(f"Failed to refresh GUI results: {error}")

    process_finished = process1 is not None and not process1.is_alive()
    if stop_thread or process_finished:
        finalize_result_monitor()
        return

    schedule_result_poll(delay_ms=500)


def export_convergence_artifacts(log_dir):
    """Save cumulative best fitness as CSV data and a PNG convergence plot."""
    samples_dir = os.path.join(log_dir, "samples")
    if not os.path.isdir(samples_dir):
        return False

    sample_files = [
        filename
        for filename in os.listdir(samples_dir)
        if filename.startswith("samples_")
        and filename.endswith(".json")
        and filename != "samples_best.json"
    ]
    if not sample_files:
        return False

    def sample_file_start(filename):
        try:
            return int(filename.removeprefix("samples_").split("~", 1)[0])
        except (TypeError, ValueError):
            return float("inf")

    samples = []
    for filename in sorted(sample_files, key=sample_file_start):
        path = os.path.join(samples_dir, filename)
        with open(path, encoding="utf-8") as file:
            file_samples = json.load(file)
        samples.extend(file_samples)

    if not samples:
        return False

    samples.sort(key=lambda sample: int(sample.get("sample_order", 0)))
    rows = []
    best_fitness = None
    for fallback_order, sample in enumerate(samples, start=1):
        sample_order = int(sample.get("sample_order", fallback_order))
        score = sample.get("score")
        valid_score = (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
        )
        fitness = float(score) if valid_score else None
        if fitness is not None and (
            best_fitness is None or fitness > best_fitness
        ):
            best_fitness = fitness
        rows.append(
            {
                "sample_order": sample_order,
                "fitness": fitness,
                "best_fitness": best_fitness,
            }
        )

    csv_path = os.path.join(log_dir, "convergence_data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("sample_order", "fitness", "best_fitness"),
        )
        writer.writeheader()
        writer.writerows(rows)

    figure = Figure(figsize=(8, 5), dpi=150)
    axis = figure.add_subplot(111)
    valid_rows = [row for row in rows if row["best_fitness"] is not None]
    if valid_rows:
        axis.plot(
            [row["sample_order"] for row in valid_rows],
            [row["best_fitness"] for row in valid_rows],
            color="tab:blue",
            linewidth=1.8,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No valid fitness values",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_title("Convergence Curve")
    axis.set_xlabel("Samples")
    axis.set_ylabel("Best Fitness")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    png_path = os.path.join(log_dir, "convergence_curve.png")
    figure.savefig(png_path)

    marker_path = os.path.join(log_dir, "llamea_output_dir.txt")
    if os.path.isfile(marker_path):
        with open(marker_path, encoding="utf-8") as file:
            llamea_log_dir = file.read().strip()
        if llamea_log_dir and os.path.isdir(llamea_log_dir):
            shutil.copy2(csv_path, os.path.join(llamea_log_dir, "convergence_data.csv"))
            shutil.copy2(png_path, os.path.join(llamea_log_dir, "convergence_curve.png"))
    return True


def plot_fig(index, log_dir, max_sample_nums):
    global figures
    global ax
    ###############################################################
    generation = []
    best_value_list = []
    all_best_value = float('-inf')
    best_alg = None

    file_name_list = [log_dir + f'/samples/samples_{i * 200 + 1}~{(i + 1) * 200}.json' for i in range(((index - 1) // 200) + 1)]

    data = []
    for file_name in file_name_list:
        with open(file_name) as file:
            data.append(json.load(file))

    for i in range(index):
        individual = data[i // 200][((i+1) % 200)-1]
        code = individual['function']
        # alg = individual['algorithm']
        obj = individual['score']
        if obj is None:
            generation.append(i + 1)
            best_value_list.append(all_best_value)
            continue
        if obj > all_best_value:
            all_best_value = obj
            best_alg = code
        generation.append(i + 1)
        best_value_list.append(all_best_value)

    generation = np.array(generation)
    best_value_list = np.array(best_value_list)

    ###############################################################
    # plot

    font = {
        'family': 'Times New Roman',
        'size': 16
    }

    figures.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.set_title(f"Result display", fontdict=font)

    # ax.plot(generation, best_value_list, color='tab:blue', marker='o')
    ax.plot(generation, best_value_list, color='tab:blue')
    ax.set_xlabel('Samples', fontdict=font)
    ax.set_ylabel('Current best objective', fontdict=font)
    ax.grid(True)

    if len(generation) <= max_sample_nums:
        if max_sample_nums<=20:
            ax.set_xticks(np.arange(0, max_sample_nums + 1, 1))
        else:
            ticks = np.linspace(0, max_sample_nums, 11)
            ticks = np.round(ticks).astype(int)
            ax.set_xticks(ticks)
    else:
        if len(generation)<=20:
            ax.set_xticks(np.arange(0, len(generation) + 1, 1))
        else:
            ticks = np.linspace(0, len(generation), 11)
            ticks = np.round(ticks).astype(int)
            ax.set_xticks(ticks)

    ###############################################################

    return figures, best_alg, all_best_value

def display_plot(index):
    global canvas
    canvas.draw()

    value_label.config(text=f"{index + 1} samples")

######################################################################

def display_alg(alg):
    code_display.config(state='normal')
    code_display.delete(1.0, 'end')
    code_display.insert(tk.END, alg)
    code_display.config(state='disabled')


def except_error():
    global process1
    try:
        if process1.exitcode == 1:
            return True
        else:
            return False
    except:
        return False


def check_finish(log_dir, index, max_sample_nums):
    return os.path.exists(log_dir + '/population/' + 'end.json') or index > max_sample_nums


def check(index, log_dir):
    temp_var1 = (index - 1) // 200
    return_value = False
    file_name = log_dir + f'/samples/samples_{temp_var1*200+1}~{(temp_var1+1)*200}.json'

    if os.path.exists(file_name):
        with open(file_name) as file:
            data = json.load(file)
        if len(data) >= ((index-1) % 200)+1:
            return_value = True
    return return_value


def stop_run_thread():
    stop_run()

def stop_run():
    global stop_thread
    global process1

    stop_button['state'] = tk.DISABLED
    stop_thread = True
    if process1 is not None:
        if process1.is_alive():
            try:
                process1.terminate()
            except:
                pass
    if not monitor_finalized:
        schedule_result_poll(delay_ms=0)
    plot_button['state'] = tk.NORMAL


def exit_run():
    stop_run()
    if not monitor_finalized:
        finalize_result_monitor()
    root.destroy()
    sys.exit(0)


###############################################################################

if __name__ == '__main__':

    root = ttk.Window()
    root.title("LLM4AD")
    root.geometry("1500x900")
    root.protocol("WM_DELETE_WINDOW", exit_run)

    root.iconbitmap(_asset_path('image', 'icon.ico'))

    style = tkttk.Style()
    style.configure("TLabelframe.Label", font=('Helvetica', 15))
    style.configure("TLabel", font=('Comic Sans MS', 12))
    style.configure("TCombobox", font=('Comic Sans MS', 10))

    photo_doc = tk.PhotoImage(file=_asset_path('image', 'document.png'))
    photoimage_doc = photo_doc.subsample(10, 10)
    photo_web = tk.PhotoImage(file=_asset_path('image', 'website.png'))
    photoimage_web = photo_web.subsample(10, 10)
    photo_git = tk.PhotoImage(file=_asset_path('image', 'github.png'))
    photoimage_git = photo_git.subsample(10, 10)
    photo_qq = tk.PhotoImage(file=_asset_path('image', 'qq.png'))
    photoimage_qq = photo_qq.subsample(10, 10)

    top_frame = ttk.Frame(root, height=30, bootstyle="info")
    top_frame.pack(fill='x')
    link_doc = ttk.Button(top_frame, image=photoimage_doc, command=open_doc_link, bootstyle="info")
    link_doc.pack(side=tk.LEFT, padx=3)
    link_git = ttk.Button(top_frame, image=photoimage_git, command=open_github_link, bootstyle="info")
    link_git.pack(side=tk.LEFT, padx=3)
    link_web = ttk.Button(top_frame, image=photoimage_web, command=open_website_link, bootstyle="info")
    link_web.pack(side=tk.LEFT, padx=3)
    link_qq = ttk.Button(top_frame, image=photoimage_qq, command=open_qq_link, bootstyle="info")
    link_qq.pack(side=tk.LEFT, padx=3)

    bottom_frame = ttk.Frame(root)
    bottom_frame.pack(fill='both', expand=True)
    left_frame = ttk.Frame(bottom_frame)
    left_frame.grid(row=0, column=0, sticky="nsew")
    ttk.Separator(bottom_frame, orient='vertical', bootstyle="secondary").grid(row=0, column=1, sticky="ns")
    right_frame = ttk.Frame(bottom_frame)
    right_frame.grid(row=0, column=2, sticky="nsew")

    bottom_frame.grid_rowconfigure(0, weight=1)
    bottom_frame.grid_columnconfigure(0, weight=2)
    bottom_frame.grid_columnconfigure(1, weight=1)
    bottom_frame.grid_columnconfigure(2, weight=30)

    #####################################################

    llm_frame = ttk.Labelframe(left_frame, text="LLM setups", bootstyle="dark")
    llm_frame.pack(anchor=tk.NW, fill=tk.X, padx=5, pady=5)

    for i in range(len(llm_para_value_name_list)):
        llm_para_entry_list.append(PlaceholderEntry(llm_frame, width=70, bootstyle="dark", placeholder=llm_para_placeholder_list[i]))
        if i != 0:
            ttk.Label(llm_frame, text=llm_para_value_name_list[i] + ':').grid(row=i - 1, column=0, sticky='ns', padx=5, pady=5)
            llm_para_entry_list[-1].grid(row=i - 1, column=1, sticky='ns', padx=5, pady=5)
            llm_frame.grid_rowconfigure(i - 1, weight=1)

    llm_frame.grid_columnconfigure(0, weight=1)
    llm_frame.grid_columnconfigure(1, weight=1)

    for i, default_value in enumerate(llm_para_default_value_list):
        if default_value:
            llm_para_entry_list[i].delete(0, 'end')
            llm_para_entry_list[i].configure(foreground=llm_para_entry_list[i].default_fg_color)
            llm_para_entry_list[i].insert(0, str(default_value))
            llm_para_entry_list[i].have_content = True

    ############

    container_frame_1 = tk.Frame(left_frame)
    container_frame_1.pack(fill=tk.BOTH, expand=True)

    algo_frame = ttk.Labelframe(container_frame_1, text="Methods", bootstyle="primary")
    algo_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    problem_frame = ttk.Labelframe(container_frame_1, text="Tasks", bootstyle="warning")
    problem_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    ############

    container_frame_2 = tk.Frame(left_frame)
    container_frame_2.pack(fill=tk.BOTH, expand=True)

    algo_param_frame = ttk.Labelframe(container_frame_2, text="eoh", bootstyle="primary")
    algo_param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    problem_param_frame = ttk.Labelframe(container_frame_2, text="admissible_set", bootstyle="warning")
    problem_param_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    ############

    algo_listbox = tk.Listbox(algo_frame, height=6, bg='white', selectbackground='lightgray', font=('Comic Sans MS', 12))
    algo_listbox.pack(anchor=tk.NW, fill='both', expand=True, padx=5, pady=5)
    default_method_index = None
    path = METHOD_DIR
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and name != '__pycache__':
            algo_listbox.insert(tk.END, name)
        if name == default_method:
            default_method_index = algo_listbox.size() - 1

    algo_listbox.bind("<<ListboxSelect>>", on_algo_select)
    on_algo_select(algo_listbox.select_set(default_method_index))

    ############

    objectives_var = tk.StringVar(value="optimization")
    objectives_frame = tk.Frame(problem_frame, bg='white')
    objectives_frame.pack(anchor=tk.NW, pady=5)
    radiobutton_list = []
    for _, dict_name, _ in os.walk(TASK_DIR):
        for name in dict_name:
            if name != '__pycache__' and name != '_data':
                radiobutton_list.append(name)
        break
    combobox = ttk.Combobox(objectives_frame, state='readonly', values=radiobutton_list, textvariable=objectives_var, bootstyle="warning", font=('Comic Sans MS', 12))
    combobox.bind('<<ComboboxSelected>>', problem_type_select)
    combobox.pack(anchor=tk.NW, padx=5, pady=5)
    problem_type_select()

    ############

    formal_save_var = tk.BooleanVar(value=True)
    save_checkbox = ttk.Checkbutton(
        left_frame,
        text="正式保存",
        variable=formal_save_var,
        bootstyle="success-round-toggle",
    )
    save_checkbox.pack(side='left', padx=5, pady=20)

    plot_button = ttk.Button(left_frame, text="Run", command=on_plot_button_click, width=12, bootstyle="primary-outline", state=tk.NORMAL)
    plot_button.pack(side='left', pady=20, expand=True)

    stop_button = ttk.Button(left_frame, text="Stop", command=stop_run_thread, width=12, bootstyle="warning-outline", state=tk.DISABLED)
    stop_button.pack(side='left', pady=20, expand=True)

    doc_button = ttk.Button(left_frame, text="Log files", command=open_folder, width=12, bootstyle="dark-outline", state=tk.DISABLED)
    doc_button.pack(side='left', pady=20, expand=True)

    ##########################################################

    state_frame = ttk.Frame(right_frame)
    state_frame.grid(row=0, column=0, sticky='ns', padx=5, pady=5)
    code_frame = ttk.Frame(right_frame)
    code_frame.grid(row=0, column=1, sticky='ns', padx=5, pady=5)

    plot_frame = tk.Frame(right_frame, bg='white')
    plot_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)

    right_frame.grid_rowconfigure(0, weight=400)
    right_frame.grid_rowconfigure(1, weight=2500)
    right_frame.grid_columnconfigure(0, weight=500)
    right_frame.grid_columnconfigure(1, weight=500)

    ###

    right_frame_label = ttk.Label(state_frame, text="Wait", anchor='w')
    right_frame_label.pack(fill=tk.X, padx=10, pady=10)

    value_label = ttk.Label(state_frame, text="0 samples", anchor='w')
    value_label.pack(fill=tk.X, padx=10, pady=10)

    objective_label = ttk.Label(state_frame, text="Current best objective:", anchor='w')
    objective_label.pack(fill=tk.X, padx=10, pady=10)

    ###

    code_display_frame = ttk.Labelframe(code_frame, text="Current best algorithm:", bootstyle="dark")
    code_display_frame.pack(anchor=tk.NW, fill=tk.X, padx=5, pady=5)
    code_display = tk.Text(code_display_frame, height=14, width=55)
    code_display.pack(fill='both', expand=True, padx=5, pady=5)
    sorting_algorithm = ""
    code_display.insert(tk.END, sorting_algorithm)
    code_display.config(state='disabled')

    ##########################################################

    root.mainloop()

    ##########################################################
