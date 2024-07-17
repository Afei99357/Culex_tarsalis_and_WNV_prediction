import cv2
import glob

# Read image sequence
images = glob.glob("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/yearly/plot/human_wnnv_*.png")
images.sort()

# Check if there are images available
if not images:
    print("No images found. Please check the directory and file paths.")
else:
    # Read the first image to get the frame size
    first_image = cv2.imread(images[0])
    if first_image is not None:
        frame_size = (first_image.shape[1], first_image.shape[0])  # (width, height)
    else:
        raise ValueError(f"Error loading the first image: {images[0]}")

    # Initialize video writer object with the correct frame size
    output = cv2.VideoWriter('/Users/ericliao/Desktop/dissertation/proposal defense/images/wnv_human_over_years_1999_2023_test.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'), 2, frame_size)

    for image_name in images:
        img = cv2.imread(image_name)
        if img is not None:
            output.write(img)
        else:
            print(f"Error loading image: {image_name}")

    # Release video writer object
    output.release()
    print("Video file generated successfully.")
