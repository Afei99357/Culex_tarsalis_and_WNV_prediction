# Read the content of the file
with open(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/SIFT/parent_dir/gene-annotation-src/all_prot.fasta",
    "r",
) as file:
    fasta_content = file.read()

# Split the content by lines
fasta_lines = fasta_content.split("\n")

# Process the lines and remove '.m01' from the chromosome names
modified_fasta = []
for line in fasta_lines:
    if line.startswith(">"):
        # Replace '.m01' with an empty string in the header line
        modified_fasta.append(line.replace(".m01", ""))
    else:
        modified_fasta.append(line)

# Write the modified content back to the file
with open(
    "/Users/ericliao/Desktop/WNV_project_files/landscape_genetics/SIFT/parent_dir/gene-annotation-src/all_prot_modify.fasta",
    "w",
) as file:
    file.write("\n".join(modified_fasta))
