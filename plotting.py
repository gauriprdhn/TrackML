import numpy as np
import torch


class ColorWheel:
    '''Returns a consistent color when given the same object'''
    def __init__(self, colors=None, seed=44):
        if colors is None:
            import matplotlib._color_data as mcd
            self.colors = list(mcd.XKCD_COLORS.values())
        else:
            self.colors = colors
        np.random.seed(seed)
        np.random.shuffle(self.colors)
        self._original_colors = self.colors.copy()
        self.assigned_colors = {}
        
    def __call__(self, thing):
        key = id(thing)
        if key in self.assigned_colors:
            return self.assigned_colors[key]
        else:
            color = self.colors.pop()
            self.assigned_colors[key] = color
            if not(self.colors): self.colors = self._original_colors.copy()
            return color
    
    def assign(self, thing, color):
        """Assigns a specific color to a thing"""
        key = id(thing)
        self.assigned_colors[key] = color
        if color in self.colors: self.colors.remove(color)


def get_plotly_pred(event, clustering):
    import plotly.graph_objects as go
    colorwheel = ColorWheel()
    colorwheel.assign(-1, '#bfbfbf')

    data = []
    energies = event.x[:,0].numpy()
    energy_scale = 20./np.average(energies)

    for cluster_index in np.unique(clustering):
        x = event.x[clustering == cluster_index].numpy()
        energy = x[:,0]
        sizes = np.maximum(0., np.minimum(3., np.log(energy_scale*energy)))
        data.append(go.Scatter3d(
            x=x[:,7], y=x[:,5], z=x[:,6],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=sizes,
                color= colorwheel(int(cluster_index)),
                ),
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}'
                f'<br>clusterindex={cluster_index}'
                f'<br>'
                ),
            name = f'cluster_{cluster_index}'
            ))
    return data


def get_plotly_truth(event):
    import plotly.graph_objects as go
    colorwheel = ColorWheel()
    colorwheel.assign(0, '#bfbfbf')

    data = []
    energies = event.x[:,0].numpy()
    energy_scale = 20./np.average(energies)

    for cluster_index in np.unique(event.y):
        x = event.x[event.y == cluster_index].numpy()
        energy = x[:,0]
        sizes = np.maximum(0., np.minimum(3., np.log(energy_scale*energy)))
        data.append(go.Scatter3d(
            x=x[:,7], y=x[:,5], z=x[:,6],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=sizes,
                color= colorwheel(int(cluster_index)),
                ),
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}'
                f'<br>clusterindex={cluster_index}'
                f'<br>'
                ),
            name = f'cluster_{cluster_index}'
            ))
    return data


def get_plotly_clusterspace(event, cluster_space_coords, clustering=None):
    assert cluster_space_coords.size(1) == 3
    import plotly.graph_objects as go

    colorwheel = ColorWheel()
    colorwheel.assign(0, '#bfbfbf')
    colorwheel.assign(-1, '#bfbfbf')

    data = []
    energies = event.x[:,0].numpy()
    energy_scale = 20./np.average(energies)

    if clustering is None: clustering = event.y

    for cluster_index in np.unique(clustering):
        x = cluster_space_coords[clustering == cluster_index].numpy()
        energy = event.x[:,0].numpy()
        sizes = np.maximum(0., np.minimum(3., np.log(energy_scale*energy)))
        data.append(go.Scatter3d(
            x=x[:,0], y=x[:,1], z=x[:,2],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=sizes,
                color= colorwheel(int(cluster_index)),
                ),
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}'
                f'<br>clusterindex={cluster_index}'
                f'<br>'
                ),
            name = f'cluster_{cluster_index}'
            ))
    return data
