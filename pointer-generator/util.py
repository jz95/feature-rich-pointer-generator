# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import os
import re
FLAGS = tf.app.flags.FLAGS


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def load_ckpt(saver, sess, ckpt_dir="train"):
    """Load checkpoint from the ckpt_dir (if unspecified,
    this is train dir) and restore it to saver and sess,
    waiting 10 secs in the case of failure. Also returns checkpoint name."""

    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir == "eval" else None
            ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(
                ckpt_dir, latest_filename=latest_filename)
            tf.logging.info('Loading checkpoint %s',
                            ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except Exception:
            tf.logging.info(
                "Failed to load checkpoint from %s.\
                Sleeping for %i secs...", ckpt_dir, 10)
            time.sleep(10)


def loop_over_ckpt(saver, sess, ckpt_dir="train"):
    ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
    if loop_over_ckpt.firstCall:
        tmp = [f for f in os.listdir(ckpt_dir) if os.path.isfile(os.path.join(ckpt_dir, f)) and re.match(r'model\.ckpt-\d+.index', f)]
        loop_over_ckpt.ckpt_files.extend(tmp)
        loop_over_ckpt.ckpt_files.sort(reverse=True)
        loop_over_ckpt.firstCall = False

    if len(loop_over_ckpt.ckpt_files) == 0:
        tf.logging.info('offline eval finished... exit...')
        exit()

    ckpt_file = loop_over_ckpt.ckpt_files.pop()[:-6]
    saver.restore(sess, os.path.join(ckpt_dir, ckpt_file))
    tf.logging.info('Succesfully loaded model %s.' % ckpt_file)

loop_over_ckpt.firstCall = True
loop_over_ckpt.ckpt_files = []
