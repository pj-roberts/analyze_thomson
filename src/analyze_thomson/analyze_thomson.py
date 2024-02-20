## Main entry point for analyze_thomson scripts

import functions as fn
import matplotlib.pyplot as plt

def main():

    filepath = "C:\\Users\\pjrob\\Desktop\\"

    # Generate datset
    ne = 1e16
    Te = 80
    u = 0
    noise = 0
    wl,spectrum = fn.generate_dataset(ne,Te,u,noise)

    # Plot resulting spectrum
    plt.plot(wl,spectrum,'-')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    ax = plt.gca()
    ax.set_box_aspect(1)
    plt.show()


if __name__ == '__main__':
    main()
    