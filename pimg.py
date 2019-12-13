import matplotlib.pyplot as pyp
import numpy as np

def imread(file):
    # read the file as a float64, scale by 255 and cast to uint8
    filetype = file.split(".")
    if (filetype[-1] == "png"):
        return (255*pyp.imread(file)).astype(np.uint8)
    else:
        return (pyp.imread(file)).astype(np.uint8)

def nchannels(image):
    """Returns the number of channels from an input image."""
    return np.size(image[0][0])

def size(image):
    """Returns an array with 2 values containing width and height from an input image, respectively."""
    return list(reversed(image.shape[:2]))

def rgb2gray(image):
    """Returns a copy of the input image converted to greyscale using the Luma Transform."""
    # (R,G,B)*(0.299,0.587,0.114), ITU-R 601-2 luma transform
    return np.dot(image, [0.299, 0.587, 0.114]).astype(np.uint8)

def imreadgray(file):
    """Receives a file's path and returns a greyscale image. It works for both RGB and greyscale input images."""
    if len(list(file.shape)) == 2:
        return rgb2gray(file)
    else:
        return imread(file)

def imshow(image):
    """Shows an image using pyplot. It the image is in greyscale, shows a grey colormap."""
    if len(list(image.shape)) == 2:
        pyp.imshow(image, cmap='gray', interpolation='nearest')
    else:
        pyp.imshow(image, interpolation='nearest')
    pyp.show()
    return

def thresh(image, threshold):
    """ A threshold function

    Receives an image and a threshold as params and returns an imagem in which each pixel has maximum intensity
    where the corresponding pixel from the input image has a value greater or equal than the threshold, returning
    0 otherwise.
    """

    newImage = np.copy(image)
    # with np.nditer(newImage, op_flags=['readwrite']) as it:
    #     for x in it:
    #         x[...] = 255 if x > threshold else 0
    newImage[newImage > threshold] = 255
    newImage[newImage < threshold] = 0
    return newImage


def negative(image):
    """Returns the negative of the input image"""
    #R = 255 – R, G = 255 – G, B = 255 – B
    return 255 - image


def contrast(newImage, r, m):
    """Returns an newImage following the newImage = r(f-m)+m equation."""
    r *= 1.0
    m *= 1.0
    newImage = r*(newImage-m)+m
    newImage[newImage < 0] = 0
    newImage[newImage > 255] = 255
    return newImage.astype(np.uint8)
    # for x in np.nditer(newImage, op_flags=['readwrite']):
    #     x[...] = 255 if x[...] > 255 else x[...] if x[...] > 0 else 0
    # return newImage.astype(np.uint8)


def hist(image):
    """Returns a column array where each position contains the number of piexls with each grey intesitiy. Returns 3 columns if the input image is RGB."""
    if nchannels(image) == 1:
        histogram = np.zeros((256, 1), dtype=int)
        for line in image:
            for pixel in line:
                histogram[pixel] += 1

    elif nchannels(image) == 3:
        histogram = np.zeros((256, 3), dtype=int)
        for line in image:
            for pixel in line:
                for index, sample in enumerate(pixel):
                    histogram[sample][index] += 1
    return histogram


def showhist(histogram):
    """Shows a bar graph using pyplot from a previously obtained histogram."""
    x_axis = np.array(range(0, 256))
    if histogram.shape[1] == 1:
        pyp.bar(x_axis, histogram.transpose()[
                0], color='black', align='center')
    elif histogram.shape[1] == 3:
        w = 0.3
        rgb = pyp.subplot(111)
        rgb.bar(x_axis - w, histogram.transpose()
                [0], width=w, color='red', align='center')
        rgb.bar(x_axis, histogram.transpose()[
                1], width=w, color='green', align='center')
        rgb.bar(x_axis + w, histogram.transpose()
                [2], width=w, color='blue', align='center')
        rgb.autoscale(tight=True)
##        rb = pyp.bar(x_axis, histogram.transpose()[0], color='red', align='center')
##        pyp.subplot(131).imshow(rb)
##        pyp.title('Red'), pyp.xticks([]), pyp.yticks([])
    pyp.show()
    return

def histeq(image):
    """Returns an equalized histogram from an input image."""
    newImage = np.copy(image)
    sz = size(image)
    n = sz[0]*sz[1]

    if nchannels(image) == 1:
        eqHistogram = np.transpose(hist(image))[0]
        #n = np.sum(eqHistogram)

        for nr in range(1, len(eqHistogram)):
            eqHistogram[nr] += eqHistogram[nr - 1]

        eqHistogram = (np.round((eqHistogram/n)*255)).astype(np.uint8)

        with np.nditer(newImage, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = eqHistogram[x]

    elif nchannels(image) == 3:
        eqHistogram = np.transpose(hist(image))

        for nr in range(1, len(eqHistogram[0])):
            for channel in range(3):
                # trocar por for depois
                eqHistogram[channel][nr] += eqHistogram[channel][nr-1]

        for i in range(sz[1]):
            for j in range(sz[0]):
                for sample, pixel in enumerate(image[i][j]):
                    newImage[i][j][sample] = eqHistogram[sample][pixel]

    return newImage


def blur(image):
    """Returns the input image convolved by the maskBlur mask"""
    return convolve(image, maskBlur())


def rot180(mask):
    if len(mask.shape) == 1:
        return np.flipud(mask)
    else:
        return np.rot90(mask, 2)


def maskBlur():
    mask = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    return mask*(1/16)


def seSquare3():
    return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


def seCross3():
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def extrapolate(image, mask):
    #((0,0),(1,1)) xe,xd e ye, yd
    return np.pad(image, int(np.floor(mask.shape[0]/2)), 'edge')


def convolve(image, mask):
    """Returns the input image convoluted with the input mask, using the nearest values when extrapolation is needed."""
    #padding = int(np.floor(mask.shape[0]/2))
    sz = size(image)
    mask = rot180(mask)
    extrapolatedImage = extrapolate(image, mask)
    res = np.zeros(image.shape, dtype=np.uint8)

    for x in range(sz[1]):
        for y in range(sz[0]):
            for i, a in enumerate(mask):
                for j, w in enumerate(a):
                    res[x][y] += w*extrapolatedImage[x + i][y + j]
    return res

def erode(image, mask):
    """Performs the erosion of the input image with a mask structuring element."""
    #padding = int(np.floor(mask.shape[0]/2))
    sz = size(image)
    extrapolatedImage = extrapolate(image, mask)
    res = np.zeros(image.shape, dtype=np.uint8)

    for x in range(sz[1]):
        for y in range(sz[0]):
            array = []
            for i, a in enumerate(mask):
                for j, w in enumerate(a):
                    if w != 0:
                        array.append(w*extrapolatedImage[x + i][y + j])
            res[x][y] = min(array)
    return res


def dilate(image, mask):
    """Performs the dilation of the input image with a mask structuring element."""
    #padding = int(np.floor(mask.shape[0]/2))
    sz = size(image)
    extrapolatedImage = extrapolate(image, mask)
    res = np.zeros(image.shape, dtype=np.uint8)
    mask = rot180(mask)

    for x in range(sz[1]):
        for y in range(sz[0]):
            array = []
            for i, a in enumerate(mask):
                for j, w in enumerate(a):
                    if w != 0:
                        array.append(w*extrapolatedImage[x + i][y + j])
            res[x][y] = max(array)
    return res


def neighborhood(image, coords):

    cells = []

    x = coords[0]
    y = coords[1]
    sz = size(image)

    if (x - 1 > 0):
        cells.append((x-1, y))
    if (x + 1 < sz[1]):
        cells.append((x+1, y))

    if (y-1 > 0):
        cells.append((x, y-1))
    if (y + 1 < sz[0]):
        cells.append((x, y+1))
    return cells


def label(image):
    current_label = 1
    sz = size(image)
    queue = []

    for x in range(sz[1] - 1):
        for y in range(sz[0] - 1):
            if (image[x][y] == 255):
                image[x][y] = current_label
                queue.append((x, y))
                while queue:
                    coords = queue.pop(0)
                    neighbors = neighborhood(image, coords)
                    for neighbor in neighbors:
                        if image[neighbor[0]][neighbor[1]] == 255:
                            image[neighbor[0]][neighbor[1]] = current_label
                            queue.append(neighbor)
                current_label += 1

    return image


def opening(image, mask):
    """Perfoms the erosion by a mask on the result of a dilation of the input image with the same mask."""
    return erode(dilate(image, mask), mask)


def closing(image, mask):
    """Perfoms the dilation by a mask on the result of an erosion of the input image with the same mask."""
    return dilate(erode(image, mask), mask)
