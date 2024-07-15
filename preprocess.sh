# Define the base directory
base_dir="/home/ziyangxie/Code/Data/sidewalk_data/mast3r-swin7"

# Loop from 0000 to 0009
for i in $(seq -f "%04g" 1 9); do
    dir="$base_dir/$i/ori_images"
    if [ -d "$dir" ]; then  # Check if it's a directory
        echo "Processing directory: $dir"
        python tools/export.py --image-dir "$dir"
    else
        echo "Skipping: $dir does not exist."
        ls -l "$base_dir/"  # List what's actually in the base directory
    fi
done