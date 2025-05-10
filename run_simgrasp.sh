#!/bin/bash

GPUID=0

BLENDER_BIN=/path/to/blender-2.93.3-linux-x64/blender  # This is your path.

RENDERER_ASSET_DIR=./data/assets
BLENDER_PROJ_PATH=./data/assets/material_lib_graspnet-v2.blend
SIM_LOG_DIR="./log/`date '+%Y%m%d-%H%M%S'`"

scene="pile"  # [pile, packed, single]
object_set="pile_subdiv"
material_type="specular_and_transparent" # [specular_and_transparent, diffuse, mixed]
render_frame_list="16,17,18,19"
expname="neugrasp"

GUI=0
RVIZ=0
CHOOSE="best"  # [best, random, highest]
NUM_TRIALS=200
METHOD='neugrasp'  # [neugrasp]

mycount=0
while (( $mycount < $NUM_TRIALS )); do

   $BLENDER_BIN $BLENDER_PROJ_PATH --background --python scripts/sim_grasp.py \
   -- $mycount $GPUID $expname $scene $object_set $material_type \
   $RENDERER_ASSET_DIR $SIM_LOG_DIR $render_frame_list $METHOD $GUI $RVIZ $CHOOSE

   /path/to/python ./scripts/stat_expresult.py -- $SIM_LOG_DIR \
      $expname 0
((mycount=$mycount+1));
done;

/path/to/python ./scripts/stat_expresult.py -- $SIM_LOG_DIR \
    $expname 1