from PIL import Image, ImageDraw, ImageFont

# read the 2 png files

# read the three png files
IBD_IBE_1 = Image.open(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/IBD_IBE_test/r_code/"
    "IBD_IBE_Plots.png",
    "r",
)

IBD_IBE_2 = Image.open(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/IBD_IBE_test/r_code/"
    "IBD_IBE_density_kernel.png",
    "r",
)

## get the size of the IBD_IBE_1
width_IBD_IBE_1, height_IBD_IBE_1 = IBD_IBE_1.size

## get the size of the IBD_IBE_2
width_IBD_IBE_2, height_IBD_IBE_2 = IBD_IBE_2.size

## create a new image with the size of the IBD_IBE_1 and IBD_IBE_2
new_image = Image.new(
    "RGB",
    (width_IBD_IBE_2, height_IBD_IBE_1 + height_IBD_IBE_2),
    (250, 250, 250),
)

# paste the IBD_IBE_1 on the top center of the new image
new_image.paste(IBD_IBE_1, (0, 0))

# paste the IBD_IBE_2 on the top of the new image
new_image.paste(IBD_IBE_2, (0, height_IBD_IBE_1))

# save the new image
new_image.save(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/Paper_results/IBD_IBE_test/"
    "IBD_IBE_4_Plots_combined.png"
)