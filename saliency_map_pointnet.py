import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during evaluating, recommend: 1,2,4,8 [default: 4]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', default=False, help='Whether to dump image for error case [default: False]')
parser.add_argument('--total_droppoints', type=int, default=100, help='Number of drop points [default: 100]')
parser.add_argument('--num_per_iteration', type=int, default=5,
                    help='Number of points dropped per iteration, where mod(total_droppoints,num_per_iteration) need to be 0 [default: 5]')
parser.add_argument('--lowdrop', action='store_true', default=False, help='Whether to use lowdrop, True: lowdrop, False:highdrop [default: False]')
parser.add_argument('--pred_labels', action='store_true', default=False, help='Whether to compute the gradients with the predicted labels, True: predicted labels, False: ground truth label [default: False]')
parser.add_argument('--scale_parameter', type=float, default=1, help='Scale parameter alpha [default: 1]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model)  # import network module
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
DUMP_DIR = FLAGS.dump_dir
TOTAL_DROP = FLAGS.total_droppoints
PER_ITER = FLAGS.num_per_iteration
SCALE_ALPHA = FLAGS.scale_parameter
PRED_LABELS = FLAGS.pred_labels
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')



NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
               open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def saliency_map_and_evaluate(num_votes=1):
    is_training = False

    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        if not PRED_LABELS:
            classify_loss = MODEL.get_loss_classification_only(pred, labels_pl, end_points)  #Compute loss via true labels
        else:
            labels_pre = tf.argmax(pred, axis=1, output_type=tf.int32)
            classify_loss = MODEL.get_loss_classification_only(pred, labels_pre, end_points) #Compute loss via predict labels

        #Compute the loss's gradients about the points' coordinates
        #Only classification loss is used here
        gradients=tf.gradients(classify_loss,pointclouds_pl)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'gradients':gradients}

    sali_eval_one_epoch(sess, ops, num_votes)


def sali_eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_label = np.squeeze(current_label)
        print(current_data.shape)
        #Initial adversatial data with current data
        adversarial_data = current_data

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            # Aggregating BEG
            batch_loss_sum = 0  # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES))  # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES))  # 0/1 for classes

            #Drop points in a cyclic pattern
            data_now = current_data[start_idx:end_idx, :, :]   # store the present batch data
            for num_drop in range(TOTAL_DROP//PER_ITER):
                rotated_data = provider.rotate_point_cloud_by_angle(data_now,0)
                feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
                gradients_val = sess.run(ops['gradients'],feed_dict=feed_dict)
                #Compute the saliency score
                gradients_val = np.reshape(gradients_val,(BATCH_SIZE,NUM_POINT,-1))
                point_radius = np.sqrt(np.sum(np.square(data_now),-1))
                saliency_score = -np.power(point_radius,SCALE_ALPHA)*np.sum(data_now*gradients_val,-1)

                if FLAGS.lowdrop:
                    saliency_score *=-1

                # Drop points according to the saliency score
                # Here, instead of directly deleting high/low score points, the points' coordinates are replaced with 0
                drop_index = np.argpartition(saliency_score[:,0:NUM_POINT-num_drop*PER_ITER], -PER_ITER, axis=1)
                for point_idx in range(BATCH_SIZE):
                    data_now[point_idx:point_idx + 1, 0:NUM_POINT - num_drop * PER_ITER, :] = \
                        data_now[point_idx:point_idx+1,drop_index[point_idx:point_idx+1,0:NUM_POINT - num_drop * PER_ITER],:]
                    data_now[point_idx:point_idx+1, NUM_POINT - (num_drop + 1) * PER_ITER:, :] = 0
            adversarial_data[start_idx:end_idx, :, :] = data_now
            print(batch_idx)


            for vote_idx in range(num_votes):
                # rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
                #                                                     vote_idx / float(num_votes) * np.pi * 2)
                rotated_data = provider.rotate_point_cloud_by_angle(adversarial_data[start_idx:end_idx, :, :],
                                                                    vote_idx / float(num_votes) * np.pi * 2)
                feed_dict = {ops['pointclouds_pl']: rotated_data,
                             ops['labels_pl']: current_label[start_idx:end_idx],
                             ops['is_training_pl']: is_training}
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                              feed_dict=feed_dict)
                batch_pred_sum += pred_val
                batch_pred_val = np.argmax(pred_val, 1)
                for el_idx in range(cur_batch_size):
                    batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
                batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
            # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
            # pred_val = np.argmax(batch_pred_classes, 1)
            pred_val = np.argmax(batch_pred_sum, 1)
            # Aggregating END

            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)
                fout.write('%d, %d\n' % (pred_val[i - start_idx], l))

                if pred_val[i - start_idx] != l and FLAGS.visu:  # ERROR CASE, DUMP!
                    img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
                                                                SHAPE_NAMES[pred_val[i - start_idx]])
                    img_filename = os.path.join(DUMP_DIR, img_filename)
                    output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, :, :]))
                    scipy.misc.imsave(img_filename, output_img)
                    error_cnt += 1

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX)
    with tf.Graph().as_default():
        saliency_map_and_evaluate(num_votes=1)
    LOG_FOUT.close()
