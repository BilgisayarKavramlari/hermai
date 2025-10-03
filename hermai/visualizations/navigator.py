import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from . import plots

class FigurePager:
    def __init__(self, pages, titles):
        self.pages = pages
        self.titles = titles
        self.idx = 0

        self.fig = plt.figure(figsize=(14, 8))
        
        self.plot_box = [0.07, 0.18, 0.86, 0.74]   # left, bottom, width, height
        self.ax = self.fig.add_axes(self.plot_box)

        self._add_buttons()
        self._draw()

    def _add_buttons(self):
        ax_prev = self.fig.add_axes([0.30, 0.06, 0.15, 0.05])
        ax_next = self.fig.add_axes([0.55, 0.06, 0.15, 0.05])
        self.b_prev = Button(ax_prev, "<--- Back")
        self.b_next = Button(ax_next, "Next --->")
        self.b_prev.on_clicked(lambda _e: self._update(-1))
        self.b_next.on_clicked(lambda _e: self._update(+1))

    def _draw(self):
        self.ax.remove()
        self.ax = self.fig.add_axes(self.plot_box)

        self.pages[self.idx](self.ax)

        self.ax.set_title("")
        self.fig.suptitle(self.titles[self.idx], y=0.97)

        self.fig.canvas.draw_idle()

    def _update(self, delta):
        self.idx = (self.idx + delta) % len(self.pages)
        self._draw()

def show_explanations_pager(local_exp, general_exp):
    def draw_local(ax):
        plots.plot_local_feature_importance(local_exp, ax=ax, show=False)

    def draw_global(ax):
        plots.plot_general_feature_importance(general_exp, ax=ax, show=False)

    def draw_tree(ax):
        plots.plot_surrogate_tree(general_exp, ax=ax, show=False)

    FigurePager(
        pages=[draw_local, draw_global, draw_tree],
        titles=["Local Feature Importance",
                "General Feature Importance (from Surrogate Model)",
                "Surrogate Decision Tree"]
    )
    plt.show()
