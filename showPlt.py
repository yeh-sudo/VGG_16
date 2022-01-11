import matplotlib.pyplot as plt
import matplotlib.image as img

def showplt():
    image = img.imread('./plot.png')
    plt.imshow(image)
    plt.axis('off')
    plt.show()
