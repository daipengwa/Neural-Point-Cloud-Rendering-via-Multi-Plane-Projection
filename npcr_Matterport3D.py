from network import *
import cv2, os, time, math
import glob
import scipy.io as io
from loss import *
from utils import *

is_training = False
use_viewdirection = True
renew_input = True
constant_initial = True
use_RGB = True
random_crop = True

d = 32
h = 512
w = 640
top_left_v = 0
top_left_u = 0
h_croped = 256
w_croped = 320
forward_time = 4
overlap = 32

channels_i = int(8)
channels_o = 3
channels_v = 3

gpu_id = 2
num_epoch = 30
decrease_epoch = 10
learning_rate = 0.0001
learning_rate_1 = 0.01

dataset = 'Matterport3D'
scene = '17DRP5sb8fy'
task = '%s_npcr_%s' % (dataset, scene)

dir1 = 'data/%s/%s/undistorted_color_images/' % (dataset, scene)
dir2 = 'pre_processing_results/%s/%s/reproject_results_%s/' % (dataset, scene, d)
# dir2 = '../3D_volume_all_points/Matterport_3D/pre_processing_results/reproject_results_%s/' % (d)
dir3 = 'data/%s/%s/undistorted_camera_parameters/' % (dataset, scene)
dir4 = 'pre_processing_results/%s/%s/weight_%s/' % (dataset, scene, d)
# dir4 = '../3D_volume_all_points/Matterport_3D/pre_processing_results/weight_%s/' % (d)
dir5 = 'pre_processing_results/%s/%s/point_clouds_simplified.ply' % (dataset, scene)

num_image = len(glob.glob(os.path.join(dir1, '*.jpg')))

image_names_train, index_names_train, camera_names_train, \
image_names_test, index_names_test, camera_names_test, \
index_names_1_train, index_names_1_test = prepare_data_matterport(dir1, dir2, dir3, dir4)


# load point clouds information
point_clouds, point_clouds_colors = loadfile(dir5)
num_points = point_clouds.shape[1]

# initial descriptor
descriptors = np.random.normal(0, 1, (1, num_points, channels_i))

if os.path.isfile('%s/descriptor.mat' % task):
    content = io.loadmat('%s/descriptor.mat' % task)
    descriptors = content['descriptors']
    print('loaded descriptors.')
else:
    if constant_initial:
        descriptors = np.ones((1, num_points, channels_i), dtype=np.float32) * 0.5

    if use_RGB:
        descriptors[0, :, 0:3] = np.transpose(point_clouds_colors) / 255.0

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % gpu_id
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sess = tf.Session()

input1 = tf.placeholder(dtype=tf.float32, shape=[1, d, None, None, channels_i])
input2 = tf.placeholder(dtype=tf.float32, shape=[1, d, None, None, channels_v])
output = tf.placeholder(dtype=tf.float32, shape=[1, None, None, channels_o])


with tf.variable_scope(tf.get_variable_scope()):
    inputs = input1
    total_channels = channels_i

    if use_viewdirection:
        inputs = tf.concat([input1, input2], axis=4)
        total_channels = total_channels + channels_v


    color_layer, alpha, network = neural_render(input=inputs, reuse=False, use_dilation=True)
    loss, p0, p1, p2, p3, p4, p5 = VGG_loss(network, output, reuse=False)

    loss_all = loss

    # calculate gradient
    gradient = tf.gradients(loss_all, input1)

var_list_all = [var for var in tf.trainable_variables()]
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_all, var_list=var_list_all)

saver = tf.train.Saver(var_list=var_list_all, max_to_keep=1000)

sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state(task)
if ckpt:
    print('load ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

##############################################################################################
if is_training:
    print('begin training!')
    all = np.zeros(20000, dtype=float)
    cnt = 0

    for epoch in range(num_epoch):

        if epoch >= decrease_epoch:
            learning_rate_1 = 0.005

        if epoch >= decrease_epoch*2:
            learning_rate_1 = 0.001

        if os.path.isdir("%s/%04d" % (task, epoch)):
            continue

        for i in np.random.permutation(len(image_names_train)):
        # for i in range(4):
        #     continue
            st = time.time()
            image_descriptor = np.zeros([1, d, h, w, channels_i], dtype=np.float32)
            view_direction = np.zeros([1, d, h, w, channels_v], dtype=np.float32)
            input_gradient_all = np.zeros([1, d, h, w, channels_i], dtype=np.float32)
            count = np.zeros([1, d, h, w, 1], dtype=np.float32)
            camera_name = camera_names_train[i]
            index_name = index_names_train[i]
            image_name = image_names_train[i]
            index_name_1 = index_names_1_train[i]

            if not (os.path.isfile(image_name) and os.path.isfile(index_name) and os.path.isfile(index_name_1)):
                print("Missing file!")
                continue

            npzfile = np.load(index_name)
            u = npzfile['u']
            v = npzfile['v']
            n = npzfile['d']
            select_index = npzfile['select_index']
            group_belongs = npzfile['group_belongs']
            index_in_each_group = npzfile['index_in_each_group']
            distance = npzfile['distance']
            each_split_max_num = npzfile['each_split_max_num']

            # load weight
            npzfile_weight = np.load(index_name_1)
            weight = npzfile_weight['weight_average']
            distance_to_depth_min = npzfile_weight['distance_to_depth_min']

            # update weight of each point feature.
            descriptor_renew_weight = (1-distance)*(1/(1+distance_to_depth_min))

            extrinsic_matrix = camera_parameter_read(camera_name)  # camera to world
            camera_position = np.transpose(extrinsic_matrix[0:3, 3])

            max_num = np.max(each_split_max_num)
            group_descriptor = np.zeros([(max(group_belongs+1)), max_num, channels_i], dtype=np.float32)
            group_descriptor[group_belongs, index_in_each_group, :] = descriptors[0, select_index, :] * np.expand_dims(weight, axis=1)

            image_descriptor[0, n, v, u, :] = np.sum(group_descriptor, axis=1)[group_belongs, :]

            view_direction[0, n, v, u, :] = np.transpose(point_clouds[0:3, select_index]) - camera_position
            view_direction[0, n, v, u, :] = view_direction[0, n, v, u, :] / (np.tile(np.linalg.norm(view_direction[0, n, v, u, :], axis=1, keepdims=True), (1, 3)) + 1e-10)

            image_output = np.expand_dims(cv2.resize(cv2.imread(image_name, -1), (w, h)), axis=0) / 255.0

            if random_crop:

                for j in np.random.permutation(forward_time):
                    movement_v = np.random.randint(0, overlap)
                    movement_u = np.random.randint(0, overlap)

                    if j==0:
                        top_left_u = 0 + movement_u
                        top_left_v = 0 + movement_v
                    if j==1:
                        top_left_u = w_croped - movement_u
                        top_left_v = 0 + movement_v
                    if j==2:
                        top_left_u = 0 + movement_u
                        top_left_v = h_croped - movement_v
                    if j==3:
                        top_left_u = w_croped - movement_u
                        top_left_v = h_croped - movement_v


                    [_, current_loss, l1, input_gradient] = sess.run([opt, loss_all, loss, gradient],
                                                                     feed_dict={input1: image_descriptor[:, :, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :],
                                                                                input2: view_direction[:, :, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :],
                                                                                output: image_output[:, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :]
                                                                                })

                    input_gradient_all[:, :, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :] = input_gradient[0] + input_gradient_all[:, :, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :]
                    count[:, :, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :] = count[:, :, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :] + 1

                if renew_input:
                    input_gradient_all = input_gradient_all/(count+1e-10)
                    descriptors[0, select_index, :] = descriptors[0, select_index, :] - learning_rate_1 * np.expand_dims(descriptor_renew_weight, axis=1) * input_gradient_all[0, n, v, u, :]


            else:

                [_, current_loss, l1, input_gradient] = sess.run([opt, loss_all, loss, gradient],
                                                                     feed_dict={input1: image_descriptor,
                                                                                input2: view_direction,
                                                                                output: image_output
                                                                                })

                if renew_input:
                    descriptors[0, select_index, :] = descriptors[0, select_index, :] - learning_rate_1 * np.expand_dims(descriptor_renew_weight, axis=1) * input_gradient[0][0, n, v, u, :]

            all[i] = current_loss * 255.0
            cnt = cnt + 1

            print('%s %s %s %.2f %.2f %s' % (epoch, i, cnt, current_loss, np.mean(all[np.where(all)]), time.time() - st))

        os.makedirs("%s/%04d" % (task, epoch))
        saver.save(sess, "%s/model.ckpt" % (task))
        io.savemat("%s/" % task + 'descriptor.mat', {'descriptors': descriptors})

        if epoch % 5 == 0:
            saver.save(sess, "%s/%04d/model.ckpt" % (task, epoch))
            io.savemat("%s/%04d/" % (task, epoch) + 'descriptor.mat', {'descriptors': descriptors})


        for id in range(len(image_names_test)):

            top_left_v = 120
            top_left_u = 160
            st = time.time()
            image_descriptor = np.zeros([1, d, h, w, channels_i])
            view_direction = np.zeros([1, d, h, w, channels_v])
            camera_name = camera_names_test[id]
            index_name = index_names_test[id]
            index_name_1 = index_names_1_test[id]

            if not (os.path.isfile(index_name) and os.path.isfile(index_name_1)):
                print('Missingg file 1!')
                continue

            npzfile = np.load(index_name)
            u = npzfile['u']
            v = npzfile['v']
            n = npzfile['d']
            select_index = npzfile['select_index']
            group_belongs = npzfile['group_belongs']
            index_in_each_group = npzfile['index_in_each_group']
            distance = npzfile['distance']
            each_split_max_num = npzfile['each_split_max_num']

            # load weight
            npzfile_weight = np.load(index_name_1)
            weight = npzfile_weight['weight_average']
            distance_to_depth_min = npzfile_weight['distance_to_depth_min']

            extrinsic_matrix = camera_parameter_read(camera_name)  # camera to world
            camera_position = np.transpose(extrinsic_matrix[0:3, 3])

            max_num = np.max(each_split_max_num)
            group_descriptor = np.zeros([(max(group_belongs + 1)), max_num, channels_i], dtype=np.float32)
            group_descriptor[group_belongs, index_in_each_group, :] = descriptors[0, select_index, :] * np.expand_dims(weight, axis=1)

            image_descriptor[0, n, v, u, :] = np.sum(group_descriptor, axis=1)[group_belongs, :]

            view_direction[0, n, v, u, :] = np.transpose(point_clouds[0:3, select_index]) - camera_position
            view_direction[0, n, v, u, :] = view_direction[0, n, v, u, :] / (np.tile(np.linalg.norm(view_direction[0, n, v, u, :], axis=1, keepdims=True), (1, 3)) + 1e-10)

            [result] = sess.run([network], feed_dict={input1: image_descriptor[:, :, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :],
                                                      input2: view_direction[:, :, top_left_v:(top_left_v + h_croped), top_left_u:(top_left_u + w_croped), :]})

            result = np.minimum(np.maximum(result, 0.0), 1.0) * 255.0
            cv2.imwrite('%s/%04d/%06d.png' % (task, epoch, id), np.uint8(result[0, :, :, :]))

            print(time.time() - st)

else:

    output_path = "%s/Test_Result/" % (task)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for id in range(len(image_names_test)):

        st = time.time()
        image_descriptor = np.zeros([1, d, h, w, channels_i])
        view_direction = np.zeros([1, d, h, w, channels_v])
        camera_name = camera_names_test[id]
        index_name = index_names_test[id]
        index_name_1 = index_names_1_test[id]

        if not (os.path.isfile(index_name) and os.path.isfile(index_name_1)):
            print('Missingg file 1!')
            continue

        npzfile = np.load(index_name)
        u = npzfile['u']
        v = npzfile['v']
        n = npzfile['d']
        select_index = npzfile['select_index']
        group_belongs = npzfile['group_belongs']
        index_in_each_group = npzfile['index_in_each_group']
        distance = npzfile['distance']
        each_split_max_num = npzfile['each_split_max_num']

        # load weight
        npzfile_weight = np.load(index_name_1)
        weight = npzfile_weight['weight_average']
        distance_to_depth_min = npzfile_weight['distance_to_depth_min']

        extrinsic_matrix = camera_parameter_read(camera_name)  # camera to world
        camera_position = np.transpose(extrinsic_matrix[0:3, 3])

        max_num = np.max(each_split_max_num)
        group_descriptor = np.zeros([(max(group_belongs + 1)), max_num, channels_i], dtype=np.float32)
        group_descriptor[group_belongs, index_in_each_group, :] = descriptors[0, select_index, :] * np.expand_dims(weight, axis=1)

        image_descriptor[0, n, v, u, :] = np.sum(group_descriptor, axis=1)[group_belongs, :]

        view_direction[0, n, v, u, :] = np.transpose(point_clouds[0:3, select_index]) - camera_position
        view_direction[0, n, v, u, :] = view_direction[0, n, v, u, :] / (np.tile(np.linalg.norm(view_direction[0, n, v, u, :], axis=1, keepdims=True), (1, 3)) + 1e-10)

        [result] = sess.run([network], feed_dict={input1: image_descriptor, input2: view_direction})

        result = np.minimum(np.maximum(result, 0.0), 1.0) * 255.0
        cv2.imwrite(output_path + '%06d.png' % id, np.uint8(result[0, :, :, :]))

        print(time.time() - st)

if __name__ == '__main__':
    pass
