from PIL import Image, ImageEnhance
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

def check_surroundings(img, x, y):
    """
    Given a pixel, check all surrounding pixels to determine if they are the same color.
    If the color of the surrounding pixels does not match, change its value to match surroundings.
    
    Args:
        img: The image you wish to edit
        x: x-coordinate of pixel of interest
        y: y-coordinate of pixel of interest
    Returns:
        True: After check surroundings has been completed
    """
    width, height = img.size
    surrounding_coords = [
        (x-1, y-1), (x, y-1), (x+1, y-1),
        (x-1, y),             (x+1, y),
        (x-1, y+1), (x, y+1), (x+1, y+1)
    ]
    
    # Get the first surrounding pixel value to compare with others
    for coord in surrounding_coords:
        if 0 <= coord[0] < width and 0 <= coord[1] < height:
            first_pixel_value = img.getpixel(coord)
            break
    else:
        return False  # If no valid surrounding pixels found
    
    # Check if all surrounding pixels have the same value
    for coord in surrounding_coords:
        if 0 <= coord[0] < width and 0 <= coord[1] < height:
            if img.getpixel(coord) != first_pixel_value:
                return False
    return True


def makebinary(img):
    """
    Function to loop through all pixels in an image and convert it to binary.
    
    Args:
        img: The image you wish to edit
    Returns:
        pixels: Final thresholded pixels of binary image
    """
    # Take tiff image and assess if each pixel is substrate (0) or material (1) to create a black and white image
    pixels = list(img.getdata())
    pix_array = np.array(img)
    # find an average value for substrate and material
    # split into top and bottom to address contrast issues
    top_mat = pix_array[80:120, 150:190]
    bot_mat = pix_array[250:300, 260:300]
    top_sub = pix_array[10:50, 80:120]
    bot_sub = pix_array[150:180, 0:25]
    avg_mat_top = np.mean(top_mat)
    avg_mat_bot = np.mean(bot_mat)
    avg_sub_top = np.mean(top_sub)
    avg_sub_bot = np.mean(bot_sub)
    # loop through values and assign them either 0 for substrate or 1 for material
    for i in range(len(pixels)):
        if i <= 45000:
            diff_sub = abs(pixels[i] - avg_sub_top)
            diff_mat = abs(pixels[i] - avg_mat_top)
            if diff_sub <= diff_mat:
                pixels[i] = 0
            else:
                pixels[i] = 1
        if i > 45000:        
            diff_sub = abs(pixels[i] - avg_sub_bot)
            diff_mat = abs(pixels[i] - avg_mat_bot)
            if diff_sub <= diff_mat:
                pixels[i] = 0
            else:
                pixels[i] = 1
    return pixels


def getGT(filename):
    """
    Function to take grayscale image of sample to determine ground truth data for classification.
    
    Args:
        filename: The data file you wish to determine the ground truth for
    Returns:
        final_pixels: The ground truth value (1 or 0) for each pixel in an array
    """
    # 0: substrate
    # 1: material

    # Parameters for cropping image (Determined manually from material of interest)
    x_start, x_end = 216, 516
    y_start, y_end = 675, 975

    # Load image
    try_img = tiff.imread(filename)
    data = try_img.transpose(1, 2, 0)

    # Crop image
    data_sub = data[y_start:y_end, x_start:x_end]

    # Convert to grayscale
    if data_sub.ndim == 3:
        # Convert to grayscale by averaging the channels
        image = np.mean(data_sub, axis=2)
        # Normalize the pixel values to the range 0-255
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # Convert back to an image
    gray_image = Image.fromarray(image)

    # Increase contrast to help with splitting between substrate and material
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2.0)
    #gray_image.show()
    #enhanced_image.show()

    # Find dimensions of image
    width, height = enhanced_image.size

    # Make image into list of 1s and 0s to indicate substrate or material
    new_pixels = makebinary(enhanced_image)

    # Plot the filtered data to test and compare to original image (0 = subsrate = black, 1 = material = white)
    new_img = Image.new('L', (width, height))
    pixel_data_mapped = [255 if pixel == 1 else 0 for pixel in new_pixels]
    new_img.putdata(pixel_data_mapped)
    #new_img.show()

    # Eliminate isolated pixels of the incorrect color
    # Improves accuracy of designation
    check_pixels = new_img.load()
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if check_surroundings(new_img, x, y):
                surrounding_pixel_value = new_img.getpixel((x-1, y-1))  # Get the color of one surrounding pixel
                if check_pixels[x, y] != surrounding_pixel_value:
                    check_pixels[x, y] = surrounding_pixel_value
                    
    # Put data back into list of 1s and 0s
    final_pixels = makebinary(new_img)

    return final_pixels