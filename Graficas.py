# -*- coding: utf-8 -*-
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class Canvas():
    def __init__(self,
                 x_puntos,
                 y_puntos,
                 label_x,
                 label_y,
                 width=369,
                 height=549, 
                 dpi=100):
        self.index = 0
        fig =  plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot()
        self.axes.scatter(x_puntos, y_puntos, zs=self.altura, zdir='y',c='#Efb810', label=(label_y + label_y))
        # Make legend, set axes limits and labels
        self.axes.legend()
        self.axes.set_xlabel(label_x)
        self.axes.set_ylabel(label_y)
        
    def get_cmap(self, n, name='hsv'):
        '''return distint color based in n'''
        return plt.cm.get_cmap(name, n)

    def add_scatter(self,
                    x_puntos,
                    y_puntos,
                    label_x,
                    label_y,):
        self.index += 1
        self.axes.scatter(x_puntos, 
                          y_puntos, 
                          zs=self.altura, 
                          zdir='y',
                          c=self.get_cmap(self.index), 
                          label=(label_y + label_y))

    def show_plot(self):
        plt.show()