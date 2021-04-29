#！/bin/bash
let i=1
path=./
cd ${path}
for file in *.json
do
     labelme_json_to_dataset ${file}
     base=$(echo ${file} | sed 's/\.json//g')
     mv ${base}_json/label.png ${i}_label.png
     mv ${base}_json/img.png ${i}_img.png
     let i=i+1
 done
