# eval depthtrack
#cd Depthtrack_workspace
#vot evaluate --workspace ./ untrack_deep
#vot analysis --nocache --name untrack_deep
#cd ..

## eval vot22-rgbd
cd VOT22RGBD_workspace
vot evaluate --workspace ./ untrack_deep
vot analysis --nocache --name untrack_deep
cd ..

