from loss.custom_vgg16 import *
import tensorflow as tf
from utils.configs import *
import matplotlib.pyplot as plt


"""#####################################################################################################################
This class calculates the losses in the output images of the network.
#####################################################################################################################"""

class cal_loss(object):

    def __init__(self, img, gt, vgg_path, sess, withtv=False):
        self.data_dict = loadWeightsData(vgg_path)

        """Build Perceptual Losses"""
        with tf.name_scope(name=config.model.loss_model + "_run_vgg16"):
            # content target feature
            vgg_c = custom_Vgg16(gt, data_dict=self.data_dict)
            fe_generated = [vgg_c.conv1_1, vgg_c.conv2_1, vgg_c.conv3_1, vgg_c.conv4_1, vgg_c.conv5_1]
            #fe_generated = [vgg_c.conv1_1, vgg_c.conv2_1]

            # feature after transformation
            vgg = custom_Vgg16(img, data_dict=self.data_dict)
            fe_input = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]
            #fe_input = [vgg.conv1_1, vgg.conv2_1]

            """# debug
            ''' restore vars '''
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            variables_to_restore = []
            for v in tf.trainable_variables():
                if not (v.name.startswith(config.model.loss_model)) and not (
                v.name.startswith('ft_merger/res_c7')) and not (v.name.startswith('ft_merger/res_c8')):
                    variables_to_restore.append(v)

            saver_h = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
            model_ckp = config.model.ckp_path_ft + config.model.ckp_lev_scale + '4/' + 'res3/'
            ckpt = tf.train.get_checkpoint_state(model_ckp)
            if ckpt and ckpt.model_checkpoint_path:
                full_path = tf.train.latest_checkpoint(model_ckp)
                saver_h.restore(sess, full_path)
            aa, bb = sess.run([fe_input, fe_generated])
            # [bs(4), 256, 256, 64]
            # [bs(4), 128, 128, 128]
            # [bs(4), 64, 64, 256]
            # [bs(4), 32, 32, 512]
            # [bs(4), 16, 16, 512]

            for i in range(config.train.batch_size_ft):
                for j in range(10):
                    plt.figure(0)
                    plt.subplot(231)
                    plt.imshow(aa[0][i, :, :, j], cmap='gray')
                    plt.subplot(232)
                    plt.imshow(bb[0][i, :, :, j], cmap='gray')
                    plt.subplot(233)
                    plt.imshow(aa[1][i, :, :, j], cmap='gray')
                    plt.subplot(234)
                    plt.imshow(bb[1][i, :, :, j], cmap='gray')
                    plt.subplot(235)
                    plt.imshow(aa[2][i, :, :, j], cmap='gray')
                    plt.subplot(236)
                    plt.imshow(bb[2][i, :, :, j], cmap='gray')
                    plt.show()
            """

        with tf.name_scope(name=config.model.loss_model + "_cal_content_L"):
            # compute feature loss
            loss_f = 0
            # for f_g, f_i in zip(fe_generated, fe_input):
            for f_g, f_i in zip(fe_generated, fe_input):
                '''## debug
                fg, fi, s = sess.run([f_g, f_i, size])
                for index in range(0, 64):
                    plt.figure(1)
                    plt.subplot(211)
                    plt.imshow(fg[0, :, :, index], cmap='gray')
                    plt.subplot(212)
                    plt.imshow(fi[0, :, :, index], cmap='gray')
                    plt.show()
                '''
                loss_f += tf.reduce_mean(tf.abs(f_g - f_i))
                # loss_f += tf.reduce_sum(tf.abs(f_g - f_i))
        self.loss_f = loss_f

        self.loss_tv = 0
        if withtv:
            """Build Total Variation Loss"""
            shape = tf.shape(img)
            height = shape[1]
            width = shape[2]
            y = tf.slice(img, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(img, [0, 1, 0, 0], [-1, -1, -1, -1])
            x = tf.slice(img, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(img, [0, 0, 1, 0], [-1, -1, -1, -1])
            # self.loss_tv = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
            self.loss_tv = tf.reduce_sum(tf.abs(x)) / tf.to_float(tf.size(x)) + tf.reduce_sum(tf.abs(y)) / tf.to_float(tf.size(y))

        # total loss
        self.loss = self.loss_f + self.loss_tv



