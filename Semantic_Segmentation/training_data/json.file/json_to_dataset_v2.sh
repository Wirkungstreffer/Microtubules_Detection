#！/bin/bash
let i=1
path=./
cd ${path}
for file in *.json
do
     labelme_json_to_dataset ${file}
     base=$(echo ${file} | sed 's/\.json//g')
     mv ${base}_json/label.png ${base}_label.png
     mv ${base}_json/label_viz.png ${base}_label_viz.png
     mv ${base}_json/img.png ${base}_image.png
     let i=i+1
 done
