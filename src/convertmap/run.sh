srcfoldername=$(yad --width=800 --height=400 --title="Select source map folder" --file-selection --directory)
foldername=$(yad --width=400 --title="Enter target folder name" --text="input here:" --entry --entry-text="tmpfolder")

mkdir input
mkdir output

cp ${srcfoldername}/data_list.json ./input
cp ${srcfoldername}/node_dict.json ./input
cp ${srcfoldername}/state_dict.json ./input

python3 1gen_topomap.py
python 2gen_states.py

mkdir ../../map/${foldername}
cp -r ./output/* ../../map/${foldername}
cp ./input/data_list.json ../../map/${foldername}/mapping_data_list.json
cp ${srcfoldername}/globalvecs.npy ../../map/${foldername}
cp -r ${srcfoldername}/images/ ../../map/${foldername}
echo "Map converted done!"
