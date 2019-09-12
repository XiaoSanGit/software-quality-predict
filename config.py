import time,os
import tensorflow as tf



time_suffix = time.strftime('%H%M%S_%Y%m%d')
save_path_prefix = 'summary_model/exp_{}'.format(time_suffix)

hparams = tf.contrib.training.HParams(
    train_dataset="./dataset/train/",
    val_dataset="./dataset/val/",
    summary_path = save_path_prefix,
    logger_name = None,
#########  about input  #############
    module_features_len = 20,
    batch_size = 2,
    n_epochs=50,
    epoch_iters=50,
    init_lr=1e-4,
    lambda_update_rate=0.5,
    end_lr=5e-6,

)
