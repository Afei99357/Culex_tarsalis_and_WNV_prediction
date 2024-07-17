import cv2
import glob

# Read image sequence
images = glob.glob("/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/California_yearly_case_distribution/human_wnv_*.png")
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
    output = cv2.VideoWriter('/Users/ericliao/Desktop/WNV_project_files/WNV/california/new_data_2004_2023/result/plots/wnv_human_over_years_CA_2003_2023_test.avi',
                             cv2.VideoWriter_fourcc(*'XVID'), 1.5, frame_size)

    for image_name in images:
        img = cv2.imread(image_name)
        if img is not None:
            output.write(img)
        else:
            print(f"Error loading image: {image_name}")

    # Release video writer object
    output.release()
    print("Video file generated successfully.")
