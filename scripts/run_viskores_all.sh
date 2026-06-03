#!/bin/bash

FEATURE_FILE="data/features/viskores_features.csv"
EXECUTABLE="viskores_feature_extraction/build/ExtractTumorFeatures"

echo "file,class,isosurface_points,isosurface_cells" > $FEATURE_FILE

for CLASS in glioma meningioma notumor pituitary
do
  for FILE in data/vtk/$CLASS/*.vtk
  do
    $EXECUTABLE "$FILE" "$CLASS" "$FEATURE_FILE"
  done
done

echo "All Viskores features extracted."
