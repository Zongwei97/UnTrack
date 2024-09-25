# Training ViPT
python tracking/train.py --script untrack --config deep_rgbx --save_dir ./output_x --mode single
#python tracking/train.py --script vipt --config deep_rgbt --save_dir ./outputt --mode multiple
#python tracking/train.py --script vipt --config deep_rgbe --save_dir ./outpute --mode multiple


# Training ViPT-shaw
#python tracking/train.py --script vipt --config shaw_rgbd --save_dir ./output --mode multiple --nproc_per_node 2
#python tracking/train.py --script vipt --config shaw_rgbt --save_dir ./output --mode multiple --nproc_per_node 2
#python tracking/train.py --script vipt --config shaw_rgbe --save_dir ./output --mode multiple --nproc_per_node 2
