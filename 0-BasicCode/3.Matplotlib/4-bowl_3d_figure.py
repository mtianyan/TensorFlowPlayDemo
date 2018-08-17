from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig, ax1 = plt.subplots(figsize=(8, 5),
                        subplot_kw={'projection': '3d'})

alpha = 0.8
r = np.linspace(-alpha,alpha,100)
X,Y= np.meshgrid(r,r)
l = 1./(1+np.exp(-(X**2+Y**2)))

ax1.plot_wireframe(X,Y,l)
ax1.plot_surface(X,Y,l, cmap=plt.get_cmap("rainbow"))
ax1.set_title("Bowl shape")

plt.show()
