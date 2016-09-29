import matplotlib.pyplot as plt


def imshow_ax(img, ax):
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)


def imshow(img):
    f, ax = plt.subplots(1)
    imshow_ax(img, ax)
    plt.xticks([]); plt.yticks([])
    f.show()


def imshow_pair(img1, img2):
    f, (ax1, ax2) = plt.subplots(1, 2)

    imshow_ax(img1, ax1)
    imshow_ax(img2, ax2)
    plt.xticks([]); plt.yticks([])
    f.show()
