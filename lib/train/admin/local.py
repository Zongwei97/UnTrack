class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/zwu/Tracking'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/zwu/Tracking/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/zwu/Tracking/pretrained_networks'
        self.got10k_val_dir = '/home/zwu/Tracking/data/got10k/val'
        self.lasot_lmdb_dir = '/home/zwu/Tracking/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/zwu/Tracking/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/home/zwu/Tracking/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/home/zwu/Tracking/data/coco_lmdb'
        self.coco_dir = '/home/zwu/Tracking/data/coco'
        self.lasot_dir = '/home/zwu/Tracking/data/lasot'
        self.got10k_dir = '/home/zwu/Tracking/data/got10k/train'
        self.trackingnet_dir = '/home/zwu/Tracking/data/trackingnet'
        self.depthtrack_dir = '/home/zwu/Tracking/data/depthtrack/train'
        self.lasher_dir = '/home/zwu/Tracking/data/lasher/trainingset'
        self.visevent_dir = '/home/zwu/Tracking/data/visevent/train'
