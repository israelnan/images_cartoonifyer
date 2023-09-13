##############################################################################
# FILE: cartoonify.py
# WRITERS: Israel Nankencki
# EXERCISE: Intro2cs2 ex6 2021-2022
# DESCRIPTION: A program for processing image to a cartoon like image.
##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
from cartoonify_helper import *
from typing import Optional
import sys


def separate_channels(image: ColoredImage) -> List[List[List[int]]]:
    """
    this function separates a colored image to its different channels.
    :param image: 3D list, contains 2D coordinates with lists of each pixel's different channels.
    :return: A 3D list with separate 2D lists for each channel in the image.
    """
    separated_list = []
    for channel in range(len(image[0][0])):
        single_channel = []
        for row in range(len(image)):
            single_row = []
            for col in range(len(image[0])):
                single_row.append(image[row][col][channel])
            single_channel.append(single_row)
        separated_list.append(single_channel)
    return separated_list


def combine_channels(channels: List[List[List[int]]]) -> ColoredImage:
    """
    this function combines back the different channels of an image after its processing.
    :param channels: A 3D list with separate 2D lists for each channel in the image.
    :return: A 3D list, contains 2D coordinates with lists of each pixel's different channels.
    """
    combined_list = []
    for row in range(len(channels[0])):
        single_row = []
        for col in range(len(channels[0][0])):
            single_pixel = []
            for channel in range(len(channels)):
                single_pixel.append(channels[channel][row][col])
            single_row.append(single_pixel)
        combined_list.append(single_row)
    return combined_list


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    """
    this function converts a colored image with RGB channels to a grayscale image.
    :param colored_image: 3D list, contains 2D coordinates with lists of each pixel's different channels.
    :return: A 2D list with each pixel grayscale value, calculated by the given formula.
    """
    grayscale_list = []
    for row in range(len(colored_image)):
        grayscale_row = []
        for col in range(len(colored_image[0])):
            grayscale_row.append(round(0.299 * colored_image[row][col][0] + 0.587 * colored_image[row][col][1]
                                       + 0.114 * colored_image[row][col][2]))
        grayscale_list.append(grayscale_row)
    return grayscale_list


def blur_kernel(size: int) -> Kernel:
    """
    this function creates a blurring kernel block according to its size.
    :param size: the squared size of the block and its denominator entrance.
    :return: A 2D squared list with the blurring entrance values.
    """
    return [[1 / (size ** 2) for _ in range(size)] for _ in range(size)]


def apply_kernel(image: SingleChannelImage, kernel: Kernel) -> SingleChannelImage:
    """
    this function is blurring a single channeled image given its set kernel.
    :param image: A 2D list with each pixel's original value.
    :param kernel: A 2D squared list with the blurring entrance values.
    :return: A 2D list with each pixel's blurred value.
    """
    blurred_image = []
    for row in range(len(image)):
        single_row = []
        for col in range(len(image[0])):
            blurred_pixel_val = blurred_pixel_calculator(image, kernel, row, col)
            if blurred_pixel_val < 0:
                single_row.append(0)
            elif blurred_pixel_val > 255:
                single_row.append(255)
            else:
                single_row.append(blurred_pixel_val)
        blurred_image.append(single_row)
    return blurred_image


def blurred_pixel_calculator(image: SingleChannelImage, kernel: Kernel, row: int, col: int) -> int:
    """
    this function helps 'apply_kernel' to calculate the given pixel blurred value.
    :param image: A 2D list with each pixel's original value.
    :param kernel: A 2D squared list with the blurring entrance values.
    :param row: A row index for the given pixel.
    :param col: A column index for the given pixel.
    :return: the blurred pixel value.
    """
    pixel_sum = 0
    for i in range(len(kernel)):
        if row + i - len(kernel) // 2 < 0 or row + i - len(kernel) // 2 > len(image) - 1:
            pixel_sum += (len(kernel) * image[row][col] * kernel[i][0])
        else:
            row_ind = row + i - len(kernel) // 2
            for j in range(len(kernel)):
                if col + j - len(kernel) // 2 < 0 or col + j - len(kernel) // 2 > len(image[0]) - 1:
                    col_ind = col
                else:
                    col_ind = col + j - len(kernel) // 2
                pixel_sum += image[row_ind][col_ind] * kernel[i][j]
    return round(pixel_sum)


def bilinear_interpolation(image: SingleChannelImage, y: float, x: float) -> int:
    """
    this function helps the resizing process by calculating a pixels' new value according the given formula.
    :param image: A 2D list with each pixels' values in a single channeled image.
    :param y: the fraction row index in between the original image coordinates.
    :param x: the fraction column index in between the original image coordinates.
    :return: the new pixels' value according to the given formula.
    """
    if y == int(y):
        row_up, row_down, dy = int(y), int(y), 1
    elif 0 < y < 1:
        row_up, row_down, dy = 0, 1, y
    else:
        row_up, row_down, dy = int(int(y) - 1), int(y), y - int(y)
    if x == int(x):
        col_left, col_right, dx = int(x), int(x), 1
    elif 0 < x < 1:
        col_left, col_right, dx = 0, 1, x
    else:
        col_left, col_right, dx = int(int(x) - 1), int(x), x - int(x)
    returned_pixel = round(image[row_up][col_left] * (1 - dy) * (1 - dx) + image[row_down][col_left] * dy * (1 - dx)
                           + image[row_up][col_right] * dx * (1 - dy) + image[row_down][col_right] * dx * dy)
    return returned_pixel


def resize(image: SingleChannelImage, new_height: int, new_width: int) -> SingleChannelImage:
    """
    this function resizes an image to the new size given.
    :param image: A 2D list with a single channel image pixel's values.
    :param new_height: the required new height for the image.
    :param new_width: the required new width for the image.
    :return: A 2D list with a single channel image pixel's values, after resizing.
    """
    new_sized_image = []
    row_relation, col_relation = ((len(image) - 1) / (new_height - 1)), ((len(image[0]) - 1) / (new_width - 1))
    for row in range(new_height):
        new_sized_row = []
        for col in range(new_width):
            if is_it_corner(new_height, new_width, row, col):
                if row != 0:
                    row_ind = len(image) - 1
                else:
                    row_ind = row
                if col != 0:
                    col_ind = len(image[0]) - 1
                else:
                    col_ind = col
                new_sized_row.append(image[row_ind][col_ind])
            else:
                new_sized_row.append(bilinear_interpolation(image, row * row_relation, col * col_relation))
        new_sized_image.append(new_sized_row)
    return new_sized_image


def is_it_corner(new_height: int, new_width: int, row: int, col: int) -> bool:
    """
    this function helps 'resize' to determine whether a given pixel is in one of the images' corners.
    :param new_height: A given height for the new sized image.
    :param new_width: A given width for the new sized image.
    :param row: A row index for the given pixel.
    :param col: A col index for the given pixel.
    :return: True if the pixel's located in on of the new image corners, False otherwise.
    """
    return (row == col == 0) or (row == new_height - 1 and col == 0) or (row == new_height - 1 and
                                                                         col == new_width - 1) or (row == 0 and col ==
                                                                                                   new_width - 1)


def scale_down_colored_image(image: ColoredImage, max_size: int) -> Optional[ColoredImage]:
    """
    this function determines whether a given image is standing with the maximum size allowed, if not it calculates the
    new sized image
    :param image: 3D list, contains 2D coordinates with lists of each pixel's different channels.
    :param max_size: the maximum size allowed for both height and width in the image.
    :return: None if the image stands with the size requirements, the legal new image if not.
    """
    if len(image) <= max_size and len(image[0]) <= max_size:
        return None
    shrunk_image = []
    separated_image = separate_channels(image)
    if len(image) > len(image[0]):
        new_height = max_size
        new_width = round((max_size / len(image)) * len(image[0]))
    else:
        new_width = max_size
        new_height = round((max_size / len(image[0])) * len(image))
    for channel in separated_image:
        shrunk_image.append(resize(channel, new_height, new_width))
    return combine_channels(shrunk_image)


def rotate_90(image: Image, direction: str) -> Image:
    """
    this function rotates an image by 90 degrees according to the notation given.
    :param image: A 2D list for a single channeled image, or a 3D list for colored image.
    :param direction: the notation wanted.
    :return: the rotated image.
    """
    if direction == 'R':
        return rotate_right(image)
    if direction == 'L':
        return rotate_left(image)


def rotate_right(image: Image) -> Image:
    """
    this function helps 'rotate_90' to rotate an image to the right.
    :param image: A 2D list for a single channeled image, or a 3D list for colored image.
    :return: the rotated image, to the right.
    """
    rotated_image = []
    for row in range(len(image[0])):
        new_row = []
        for col in range(len(image) - 1, -1, -1):
            new_row.append(image[col][row])
        rotated_image.append(new_row)
    return rotated_image


def rotate_left(image: Image) -> Image:
    """
    this function helps 'rotate_90' to rotate an image to the left.
    :param image: A 2D list for a single channeled image, or a 3D list for colored image.
    :return: the rotated image, to the left.
    """
    rotated_image = []
    for row in range(len(image[0]) - 1, -1, -1):
        new_row = []
        for col in range(len(image)):
            new_row.append(image[col][row])
        rotated_image.append(new_row)
    return rotated_image


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int, c: int) -> SingleChannelImage:
    """
    this function gets the edges of a given single channeled image.
    :param image: A 2D list with each pixels' value.
    :param blur_size: the given size for blurring the image.
    :param block_size: the size of the block for calculating the threshold value for each pixel.
    :param c: a given constant to clear that's an edge.
    :return: A 2D list with each edged pixel as black, white otherwise.
    """
    blurred_image = apply_kernel(image, blur_kernel(blur_size))
    threshold_list = threshold_calculator(blurred_image, block_size, c)
    edged_image = []
    for row in range(len(image)):
        edged_row = []
        for col in range(len(image[0])):
            if blurred_image[row][col] < threshold_list[row][col]:
                edged_row.append(0)
            else:
                edged_row.append(255)
        edged_image.append(edged_row)
    return edged_image


def threshold_calculator(blurred_image: SingleChannelImage, r: int, c: int) -> SingleChannelImage:
    """
    this function helps 'get_edges' to calculate the threshold list with each pixels' threshold value.
    :param blurred_image: A 2D list with each pixels' blurred value.
    :param r: the block size to mean any pixel on.
    :param c: a given constant to clear that's an edge.
    :return: A 2D list with each pixels' threshold value.
    """
    threshold_list = []
    for row in range(len(blurred_image)):
        threshold_row = []
        for col in range(len(blurred_image[0])):
            threshold_row.append(average_calculator(blurred_image, r, row, col) - c)
        threshold_list.append(threshold_row)
    return threshold_list


def average_calculator(image: SingleChannelImage, r: int, row: int, col: int) -> int:
    """
    this function helps 'threshold_calculator' to calculate a given pixel mean on the given block size.
    :param image: A 2D list with each pixels' blurred value.
    :param r: the block size to mean any pixel on.
    :param row: a given row index of the pixel to mean.
    :param col: a given column index of the pixel to mean.
    :return: the pixel mean on the given block size.
    """
    cells_sum = 0
    for i in range(r):
        if row + i - r // 2 < 0 or row + i - r // 2 > len(image) - 1:
            cells_sum += (r * image[row][col])
        else:
            for j in range(r):
                if col + j - r // 2 < 0 or col + j - r // 2 > len(image[0]) - 1:
                    cells_sum += image[row][col]
                else:
                    cells_sum += image[row + i - r // 2][col + j - r // 2]
    return cells_sum / (r ** 2)


def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    """
    this function is quantizing a single channeled image according to the number of shades given.
    :param image: A 2D list with the images' sharp values.
    :param N: the number of shades given.
    :return: A 2D list with all quantized pixels in the image.
    """
    quantized_image = []
    for row in range(len(image)):
        quantized_row = []
        for col in range(len(image[0])):
            quantized_row.append(round((int((image[row][col] * N) / 256) * 255) / (N - 1)))
        quantized_image.append(quantized_row)
    return quantized_image


def quantize_colored_image(image: ColoredImage, N: int) -> ColoredImage:
    """
    this function is quantizing a colored image according to the number of shades given.
    :param image: A 3D list with the images' sharp values.
    :param N: the number of shades given.
    :return: A 3D list with all quantized pixels in the image.
    """
    separated_image = separate_channels(image)
    quantized_channels = [quantize(channel, N) for channel in separated_image]
    return combine_channels(quantized_channels)


def add_mask(image1: Image, image2: Image, mask: List[List[float]]) -> Image:
    """
    this function combines two images, either both colored or single channeled, according to a given mask.
    :param image1: A 2D list for a single channeled image, or a 3D list for colored image.
    :param image2: A 2D list for a single channeled image, or a 3D list for colored image.
    :param mask: A 2D list in the same size of the two images, that all of its values are between 1 and 0 to determine
    which images' pixel of both images would be included and how much of it in the final image.
    :return: an image with combination of both images according to the given mask.
    """
    if type(image1[0][0]) is int:
        return add_mask_to_single_channel(image1, image2, mask)
    separated_image1, separated_image2 = separate_channels(image1), separate_channels(image2)
    new_channeled_image = []
    for channel in range(len(separated_image1)):
        new_channeled_image.append(add_mask_to_single_channel(separated_image1[channel], separated_image2[channel],
                                                              mask))
    return combine_channels(new_channeled_image)


def add_mask_to_single_channel(image1: Image, image2: Image, mask) -> Image:
    """
    this function helps 'add_mask' to combine both single channeled images according to the given mask.
    :param image1: A 2d list with all the image pixels' values.
    :param image2: A 2d list with all the image pixels' values.
    :param mask: A 2D list in the same size of the two images, that all of its values are between 1 and 0 to determine
    which images' pixel of both images would be included and how much of it in the final image.
    :return: an image with combination of both single channeled images according to the given mask.
    """
    new_image = []
    for row in range(len(image1)):
        new_row = []
        for col in range(len(image1[0])):
            new_row.append(round(image1[row][col] * mask[row][col] + image2[row][col] * (1 - mask[row][col])))
        new_image.append(new_row)
    return new_image


def cartoonify(image: ColoredImage, blur_size: int, th_block_size: int,
               th_c: int, quant_num_shades: int) -> ColoredImage:
    """
    this function receives a regular image and calculates its cartoon like image.
    :param image: 3D list, contains 2D coordinates with lists of each pixel's different channels.
    :param blur_size: the given size for blurring the image.
    :param th_block_size: the size of the block for calculating the threshold value for each pixel.
    :param th_c: a given constant to clear that's an edge.
    :param quant_num_shades: the number of shades given.
    :return: A 3D list with all cartoon like pixels' channels values lists.
    """
    outline = get_edges(RGB2grayscale(image), blur_size, th_block_size, th_c)
    normalized_outline = normalize_outline(outline)
    separated_image = separate_channels(image)
    quantized_separated_image = [quantize(channel, quant_num_shades) for channel in separated_image]
    masked_separated_image = [add_mask(quantized_channel, outline, normalized_outline) for quantized_channel in
                              quantized_separated_image]
    return combine_channels(masked_separated_image)


def normalize_outline(outline_image: SingleChannelImage) -> List[List[float]]:
    """
    this function normalize the outline images' pixels in order to use it as a mask for combining the cartoon like
    image.
    :param outline_image: A 2D list with all the outline pixels black, and others as white.
    :return: A 2D list with all outline pixel between 1 and 0.
    """
    normalized_outline_image = []
    for row in outline_image:
        normalized_outline_row = []
        for col in row:
            normalized_outline_row.append(col / 255)
        normalized_outline_image.append(normalized_outline_row)
    return normalized_outline_image


def is_it_valid_input(argv_list: List[str]) -> bool:
    """
    this function helps to determine whether the number of arguments received is as required or not.
    :param argv_list: list of all string arguments received from command line.
    :return: True if the number of arguments received is 7 as required, False otherwise.
    """
    if len(argv_list) == 8:
        return True
    print("this isn't the exact number of arguments we've expected")
    return False


if __name__ == '__main__':
    if not is_it_valid_input(sys.argv):
        sys.exit()
    else:
        image_file = load_image(sys.argv[1])
        if scale_down_colored_image(image_file, int(sys.argv[3])):
            image_file = scale_down_colored_image(image_file, int(sys.argv[3]))
        save_image(cartoonify(image_file, int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])),
                   sys.argv[2])
