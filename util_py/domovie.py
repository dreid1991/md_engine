import sys
sys.path.append('/home/daniel/Documents/md_engine_opt/core/python')
from Sim import *
import matplotlib.pyplot as plt
from math import *
from numpy import array
import numpy as np
if __name__ == '__main__':
    fn = sys.argv[1]





colors = ['r', 'g', 'b', 'c', 'k', (.5, .5, .5)]
def plot_atoms(ax, atoms, forceColor=None):
    max_type = max(a.type for a in atoms)
    for i in range(max_type+1):
        atoms_by_type = [a for a in atoms if a.type == i]
        if not len(atoms_by_type) and i:
            break
        else:
            xs = [a.pos[0] for a in atoms_by_type]
            ys = [a.pos[1] for a in atoms_by_type]
            color = forceColor if forceColor != None else colors[i]
            ax.scatter(xs, ys, s=10, color=color)

def bondWithOffset(ax, sides, x1, x2, y1, y2, cutoff):
    #1 stays put, 2 moves
    spanX = sides[0][0]
    spanY = sides[1][1]
    if abs(y2-y1) > cutoff:
        #print 'looping y'
        if y1 < y2:
            y2 -= spanY
        else:
            y2 += spanY
    if abs(x2-x1) > cutoff:
        if x1 < x2:
            x2 -= spanX
        else:
            x2 += spanX
    return [[x1, x2], [y1, y2]]
    ax.plot([x1, x2], [y1, y2], color='k')



def plot_bonds_stress(state, ax, bonds, stresses, colorMin, colorMax, zorder=1):
    logStresses = [log10(s) for s in stresses]
    minStress, maxStress = min(logStresses), max(logStresses)
    bounds = state.bounds
    lo = bounds.lo
    sides = [state.bounds.getSide(i) for i in range(2)]
    hi = bounds.hi
    cutoff = .8 * np.average([sides[i][i] for i in range(2)])
    for i, b in enumerate(bonds):
        #print 'new bond'
        first = b.first().pos
        second = b.second().pos
        dx = second[0]-first[0]
        dy = second[1]-first[1]
        fracStress = (logStresses[i] - minStress) / (maxStress - minStress)
        color = colorMax * fracStress + colorMin * (1 - fracStress)
        if sqrt(dx*dx + dy*dy) < cutoff:
            ax.plot([first[0], second[0]], [first[1], second[1]], color=color, zorder=zorder)
        else:
         #   print 'doing loop'
            xs, ys = bondWithOffset(ax, sides, first[0], second[0], first[1], second[1], cutoff)
            ax.plot(xs, ys, color=color)
            xs, ys = bondWithOffset(ax, sides, second[0], first[0], second[1], first[1], cutoff)
            ax.plot(xs, ys, color=color)



def plot_bonds(state, ax, bonds, color='k', zorder=1):
    bounds = state.bounds
    lo = bounds.lo
    sides = [state.bounds.getSide(i) for i in range(2)]
    hi = bounds.hi
    cutoff = .8 * np.average([sides[i][i] for i in range(2)])
    for b in bonds:
        #print 'new bond'
        first = b.first().pos
        second = b.second().pos
        dx = second[0]-first[0]
        dy = second[1]-first[1]
        if sqrt(dx*dx + dy*dy) < cutoff:
            ax.plot([first[0], second[0]], [first[1], second[1]], color=color, zorder=zorder)
        else:
         #   print 'doing loop'
            xs, ys = bondWithOffset(ax, sides, first[0], second[0], first[1], second[1], cutoff)
            ax.plot(xs, ys, color=color)
            xs, ys = bondWithOffset(ax, sides, second[0], first[0], second[1], first[1], cutoff)
            ax.plot(xs, ys, color=color)


def plot_bounds(ax, b):
    lo = b.lo
    pts = [b.lo, b.lo + b.getSide(0), b.hi, b.lo + b.getSide(1), b.lo]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color='k', linewidth=2)

def graph_frame(ax, state, cfgIdx, stresses=[], colorMin=np.array([0., 1., 0.]), colorMax=np.array([1., 0., 0.]), forceColor=None, doBonds=True):
    b = state.bounds
    plot_bounds(ax, b)

    plot_atoms(ax, state.atoms, forceColor=forceColor)
    if doBonds:
        if stresses == []:
            plot_bonds(state, ax, state.bonds)
        elif len(stresses) == len(state.bonds):
            plot_bonds_stress(state, ax, state.bonds, stresses, colorMin, colorMax)
        else:
            print 'Problem with bond stresses or something'
    #ax.set_xlim([-40, 40])
    #ax.set_ylim([-40, 40])

def domovie(fn, xlim=None, ylim=None, tag = '', forceColor=None, doBonds=True):
    state = State()
    state.readConfig.loadFile(fn)
    idx = 0
    while state.readConfig.next():
        axis = plt.gca()
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        graph_frame(axis, state, idx, forceColor=forceColor, doBonds=doBonds)
        plt.savefig('%s_%d.png' % (tag, idx))
        idx+=1
        plt.clf()
    state.destroy()

if __name__ == '__main__':
	domovie(fn, doBonds=False)

