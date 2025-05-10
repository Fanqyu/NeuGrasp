#!/bin/bash

cd /path/to/NeuGrasp/data_generator/render/renderer || exit

# 830*12
mycount=0;
while (( $mycount < 9960));  #830 * 12
do
    /path/to/blender-2.93.3-linux-x64/blender material_lib_v2.blend --background -noaudio \
    --python ../render_packed_STD_rand.py -- $mycount;
((mycount=$mycount+1));
done