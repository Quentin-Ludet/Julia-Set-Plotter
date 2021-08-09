# required libraries:
# numpy
# PIL
# multiprocessing
# timeit
# matplotlib

# constants

# max number of iterations
# for c > 2, pick a low number, like 8
# for c < 2, something between 30 and 60 usually works
N = 300

# image size in pixels
IMG_SIZE = (3000, 3000)

# all starting values to try with format:
# (x min, x max, y min, y max)
DOMAIN = (-2, 2, -2, 2)

# c value in Qc(x) with format:
# (real part, imaginary part)
C = (-0.75, 0.1)


# code
from PIL import Image
import numpy as np

#C_set = np.linspace(0,2,100)

class complex_number:
    def __init__(self, r, i):
        self.r = r
        self.i = i

    def __add__(self, other):
        return complex_number(self.r + other.r, self.i + other.i)

    def __mul__(self, other):
        r2 = self.r * other.r
        i2 = self.i * other.i
        ri = self.r * other.i
        ir = self.i * other.r
        return complex_number(r2 - i2, ri + ir)

    def __str__(self):
        return str(self.r) + ' + ' + str(self.i) + 'i'
    
    def print_array(array):
        for row in array[:,]:
            for n in row:
                print(n, end=' ')
            print()

    def __abs__(self):
        return np.sqrt(self.r**2 + self.i**2)

    def sin(num):
        return complex_number(np.sin(num.r) * np.cosh(num.i), np.cos(num.r) * np.sinh(num.i))


# max iterations


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

class color_map:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    r, g, b, a = self.scalarMap.to_rgba(val)
    return (r * 255,g * 255,b * 255)

cmap = color_map('inferno_r', 0, N)


def make_image(c, Y):
    X = np.linspace(DOMAIN[0],DOMAIN[1],IMG_SIZE[0])

    f = lambda x: x * x + c

    # escape value 
    max = np.max([abs(c), 2])
    #max = 2
    
    XY = []
    for y in Y:
        tempY = []
        for x in X:
            tempY.append(complex_number(x, y))
        XY.append(tempY)

    c_plane = np.array(XY)

    def julia(n):
        for i in range(N):
            if abs(n) > max:
                return cmap.get_rgb(i)
            n = f(n)
        return cmap.get_rgb(N)

    prog = 0
    filled_set = []
    for Y in c_plane:
        print(str(prog / len(c_plane))[:5] + '\r', end='')
        temp = []
        for n in Y:
            temp.append(julia(n))
        filled_set.insert(0,temp)
        prog += 1
    print()


    # Convert the pixels into an array using numpy
    return np.array(filled_set, dtype=np.uint8)

# Use PIL to create an image from the new array of pixels
def generate_file(img, C):
    new_image = Image.fromarray(img)
    fname = f'./julia/multiprocessed/{C[0]}+{C[1]}i.png'
    open(fname, 'w').close()
    new_image.save(fname)

from multiprocessing import Pool, cpu_count
from timeit import default_timer as timer

def make_frame(C):
    Y = np.linspace(DOMAIN[2],DOMAIN[3],IMG_SIZE[1])
    Y_parts = np.array_split(Y, cpu_count())
    c = complex_number(C[0], C[1])

    values = []
    for y in Y_parts:
        values.insert(0,(c, y))

    with Pool() as pool:
        res = pool.starmap(make_image, values)
        img = np.vstack(res)
        generate_file(img, C)

if __name__ == '__main__':

    start = timer()

    # for C in C_set:
    #     make_frame((C, 0))
    make_frame(C)

    end = timer()
    print(f'elapsed time: {end - start}s')

    