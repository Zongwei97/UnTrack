# test lasher

sh eval_rgbd.sh
CUDA_VISIBLE_DEVICES=0 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name untrack --dataset_name LasHeR --yaml_name deep_rgbt
CUDA_VISIBLE_DEVICES=0 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name untrack --dataset_name LasHeR --yaml_name deep_rgbt

# test rgbt234

CUDA_VISIBLE_DEVICES=0 python ./RGBE_workspace/test_rgbe_mgpus.py --script_name untrack --yaml_name deep_rgbe
CUDA_VISIBLE_DEVICES=0 python ./RGBE_workspace/test_rgbe_mgpus.py --script_name untrack --yaml_name deep_rgbe


CUDA_VISIBLE_DEVICES=0 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name untrack --dataset_name RGBT234 --yaml_name deep_rgbt
CUDA_VISIBLE_DEVICES=0 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name untrack --dataset_name RGBT234 --yaml_name deep_rgbt

