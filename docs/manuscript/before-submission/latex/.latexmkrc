# Configure latexmk to use a build directory
$out_dir = "build";
$aux_dir = "build";

# Ensure the build directory exists
system("mkdir -p build");

# Configure pdflatex to put synctex files in build directory
$pdflatex = "pdflatex -interaction=nonstopmode -synctex=1 -output-directory=build %O %S";

# PDF viewer settings (optional)
$pdf_previewer = "open -a Preview %O %S";

# Clean up settings
$clean_ext = "aux bbl blg fdb_latexmk fls log out synctex.gz"; 