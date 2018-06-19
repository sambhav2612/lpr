from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localise

# map all connected regions in the binary image sent
label_image = measure.label(localise.binary_car_image)

plate_size = (0.08*label_image.shape[0], 0.2*label_image.shape[0],
              0.15*label_image.shape[0], 0.4*label_image.shape[0])
min_height, max_height, min_width, max_width = plate_size
plate_object_points = []
plate_like_object = []

fig, ax1 = plt.subplots(1)
ax1.imshow(localise.gray_car_image, cmap="gray")

for region in regionprops(label_image):
    if region.area < 50:
        # skip region for being too small
        continue

    minRow, minCol, maxRow, maxCol = region.bbox  # bounding box
    region_height = maxRow-minRow
    region_width = maxCol-minCol

    if region_height >= min_height and region_height <= min_height and region_width >= min_width and region_width <= min_width and region_width > region_height:
        plate_like_object.append(
            localise.binary_car_image[minRow:maxRow, minCol:maxCol])
        plate_object_points.append((minRow, minCol, maxRow, maxCol))

        # this draws a red rectangle outside each analysed component of the license plate imported from localise.py
        rectBorder = patches.Rectangle(
            (minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)

plt.show()
