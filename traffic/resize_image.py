import cv2

# Load the image


# Load the image
def resize(img_dir):
    img = cv2.imread(img_dir)

    # Check if the image was loaded successfully
    if img is None:
        print('Error: Could not load image')
        exit()

    # Get the dimensions of the original image
    height, width = img.shape[:2]

    # Compute the center of the image
    center_x = int(width / 2)
    center_y = int(height / 2)

    # Compute the top-left and bottom-right corners of the crop region
    crop_size = 30
    left = center_x - int(crop_size / 2)
    top = center_y - int(crop_size / 2)
    right = center_x + int(crop_size / 2)
    bottom = center_y + int(crop_size / 2)

    # Crop the image
    cropped_img = img[top:bottom, left:right]

    # Resize the cropped image to 30x30
    resized_img = cv2.resize(cropped_img, (30, 30))
    return img
# Display the original and cropped/resized images side by side

img = resize("C:\working\cs50AI\week5\gtsrb\gtsrb\\9\\00000_00009.ppm")
# pre_img = cv2.imread("C:\working\cs50AI\week5\\test_images\\two_cars.ppm")
# cv2.imshow('Detected Shapes', pre_img)

cv2.imshow('Detected Shapes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()