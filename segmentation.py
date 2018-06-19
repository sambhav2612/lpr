import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cca

# convert black to white and vice versa
license_plate = np.invert(cca.plate_like_object[2])
labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")

character_size = (0.05*license_plate.shape[0], 0.15*license_plate.shape[0],
                  0.35*license_plate.shape[0], 0.6*license_plate.shape[0])
min_height, max_height, min_width, max_width = character_size

characters = []
counter = 0
column_list = []

for regions in regionprops(labelled_plate):
    y0, y1, x0, x1 = regions.bbox
    region_height = y1-y0
    region_width = x1-x0

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]

        rect_border = patches.Rectangle(
            (x0, y0), y1-y0, x1-x0, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        # take note of the starting x-axis of each region
        column_list.append(x0)

plt.show()
