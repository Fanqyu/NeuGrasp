#!/bin/bash

#cd /data/NeuGraspData/renderer/renderer_giga_GPU6-0_rand_M
cd /path/to/NeuGrasp/data_generator/render/renderer || exit

# 830*6
mycount=200;
while (( $mycount < 201));
do
#    /home/xxx/blender-2.93.3-linux-x64/blender material_lib_v2.blend --background -noaudio --python render_pile_STD_rand.py -- $mycount;
    /path/to/blender-2.93.3-linux-x64/blender material_lib_v2.blend --background -noaudio \
    --python ../render_pile_STD_rand.py -- $mycount;
((mycount=$mycount+1));
done