from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/zwu/Tracking/data/got10k_lmdb'
    settings.got10k_path = '/home/zwu/Tracking/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/zwu/Tracking/data/itb'
    settings.lasot_extension_subset_path_path = '/home/zwu/Tracking/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/zwu/Tracking/data/lasot_lmdb'
    settings.lasot_path = '/home/zwu/Tracking/data/lasot'
    settings.network_path = '/home/zwu/Tracking/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/zwu/Tracking/data/nfs'
    settings.otb_path = '/home/zwu/Tracking/data/otb'
    settings.prj_dir = '/home/zwu/Tracking'
    settings.result_plot_path = '/home/zwu/Tracking/output/test/result_plots'
    settings.results_path = '/home/zwu/Tracking/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/zwu/Tracking/output'
    settings.segmentation_path = '/home/zwu/Tracking/output/test/segmentation_results'
    settings.tc128_path = '/home/zwu/Tracking/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/zwu/Tracking/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/zwu/Tracking/data/trackingnet'
    settings.uav_path = '/home/zwu/Tracking/data/uav'
    settings.vot18_path = '/home/zwu/Tracking/data/vot2018'
    settings.vot22_path = '/home/zwu/Tracking/data/vot2022'
    settings.vot_path = '/home/zwu/Tracking/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

