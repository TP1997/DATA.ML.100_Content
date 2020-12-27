import numpy as np
import matplotlib.pyplot as plt

# Initalize the program
fig = plt.figure()
ax = fig.add_subplot(111)
xlim=(-30,30)
ylim=(-30,30)
ax.set(xlim=xlim,ylim=ylim)
ax.set_title('LEFT CLICK: Add point, RIGHT CLICK: Draw a fitted linear model.')
coords = []

def my_linfit():
    N=len(coords)
    xs=[x for (x,y) in coords]
    ys=[y for (x,y) in coords]
    sumx=np.sum(xs)                             # Get sum of x's
    sumxSq = np.sum([x*x for x in xs])          # Get sum of x^2's
    sumy=np.sum(ys)                             # Get sum of y's
    sumxy = np.sum([x*y for x, y in coords])    # Get sum of x*y's
    
    a = (N*sumxy-sumx*sumy)/(N*sumxSq - pow(sumx,2))
    b = np.mean(ys)-a*np.mean(xs)
    return a,b

def drawLine():
    if len(coords)>1:
        a,b=my_linfit()
        xp=np.arange(xlim[0],xlim[1],0.1)
        plt.plot(xp, a*xp+b, 'r-')
    else:
        print("Give at least two points. Program terminated")
    return

# Event handler for mouse. 
def onclick(event):
    # In case of right-click.
    if event.button==3: 
        fig.canvas.mpl_disconnect(cid)
        drawLine()
    # Ow collect mouse coordinates.
    else:
        click = event.xdata, event.ydata
        if None not in click:
            coords.append(click)
            plt.plot(click[0], click[1], 'ro')
    fig.canvas.draw_idle()

# Call event handler.
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

