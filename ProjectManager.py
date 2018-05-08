import numpy as np
import matplotlib
import matplotlib.pyplot as pyl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import pickle
import Tkinter as Tk
import tkFileDialog

matplotlib.use("TkAgg")

class PM:
    def __init__(self):
        self.MainWindow = Tk.Tk()
        self.MainWindow.title('Project Manager')

        self.Display = Figure(figsize=(7,7), dpi=150)
        self.DisplayAx = self.Display.add_subplot(111)

        self.DisplayCanvas = FigureCanvasTkAgg(self.Display, self.MainWindow)
        self.DisplayCanvas.show()
        self.DisplayCanvas.get_tk_widget().grid(row = 0, column = 0, rowspan = 8)


Pm = PM()
