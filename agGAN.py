from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from scipy.misc import imread, imresize, imsave
from ops import *
import time

class agGAN(object):
    """ The implementation of agGAN """
    def __init__(self,
                 session, # TensorFlow session
                 image_size = 128, # size of input image
                 kernel_size = 3, # size of kernel in convolution and deconcolution
                 batch_size = 10, # mini-batch for training 
                 num_input_channels = 3, # number of channels in input images
                 num_encoder_channels = 64, # number of channels in the first conv layer of encoder
                 num_z_channels = 320, # number of channels of feature representation
                 num_gen_channels = 2048,
                 num_categories = 10, # number of num_categories(age groups) in the training set
                 save_dir = './save', # path to save checkpoints, samples and summary
                 dataset_dir = '', # path to dataset
                 list_file = '',
                 mode = 'train'
                ):
        self.session = session
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.num_gen_channels = num_gen_channels
        self.num_categories = num_categories
        self.save_dir = save_dir
        self.dataset_dir = dataset_dir
        self.list_file = list_file
        self.mode = mode
        self.num_person = 1876

        # *********************************input to graph****************************************
        image_list = get_dataset(self.dataset_dir, self.list_file)
        assert len(image_list) > 0, 'The dataset should not be empty'
        self.data_size = len(image_list)
        print('num of images', len(image_list))
        
        with tf.name_scope('load_images'):
            path_queue = tf.train.string_input_producer(image_list, shuffle=self.mode == "train")

            #number of threads to read image
            num_preprocess_threads = 4
            images_and_labels = []
            for _ in range(num_preprocess_threads):
                row = path_queue.dequeue()
                """
                row = tf.reshape(row, [1])
                split = tf.string_split(row, delimiter=' ')
                fname = split.values[0]
                label_id = split.values[1]
                label_id = tf.string_to_number(label_id, tf.int32)
                label_age = split.values[2]
                label_age = tf.string_to_number(label_age, tf.int32)
                """
                fname, label_id, label_age = tf.decode_csv(records=row, record_defaults=[["string"], [""], [""]], field_delim=" ")
                label_id = tf.string_to_number(label_id, tf.int32)
                label_age = tf.string_to_number(label_age, tf.int32)

                #read image
                contents = tf.read_file(fname)
                decode = tf.image.decode_jpeg
                raw_input = decode(contents)
                #scale to 0~1
                raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
                image = transform(raw_input)
                images_and_labels.append([image, label_id, label_age])

            self.input_batch, self.label_id_batch, self.label_age_batch = tf.train.batch_join(images_and_labels, batch_size=self.batch_size,
                                                    shapes=[(self.image_size, self.image_size, self.num_input_channels), (), ()],
                                                    capacity=4 * num_preprocess_threads * self.batch_size, allow_smaller_final_batch=True)

        #*************************************build the graph************************************
        with tf.variable_scope('generator'):
            #encoder input image -> logits_id logits_age logits_id_classify
            self.logits_age, self.logits_id, self.logits_id_classify = self.encoder(self.input_batch)

            #decoder: z + label_age -> generated image
            self.G = self.decoder(self.logits_id, self.label_age_batch)

        with tf.variable_scope('discriminator'):
            # discriminator on input image
            self.D_input_logits_id, self.D_input_logits_age = self.discriminator(self.input_batch, is_training=self.mode=='train')

            # discriminator on G
            self.D_G_logits_id, self.D_G_logits_age = self.discriminator(self.G, is_training=self.mode=='train', reuse_variables=True)
        
        #*************************************loss function**************************************
        with tf.name_scope('total_variation_loss'):
            # total variation to smooth the generated image
            tv_y_size = self.image_size
            tv_x_size = self.image_size
            self.tv_loss = (
            (tf.nn.l2_loss(self.G[:, 1:, :, :] - self.G[:, :self.image_size - 1, :, :]) / tv_y_size) +
            (tf.nn.l2_loss(self.G[:, :, 1:, :] - self.G[:, :, :self.image_size - 1, :]) / tv_x_size)) / self.batch_size
        
        with tf.name_scope('encoder_loss'):
            self.E_loss_age = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_age_batch, logits=self.logits_age))

            self.E_loss_id = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_id_batch, logits=self.logits_id_classify))

            #loss of encoder
            self.loss_E = self.E_loss_id + self.E_loss_age

        with tf.name_scope('generator_loss'):
            self.G_loss_L1 = tf.reduce_mean(tf.abs(self.input_batch - self.G)) #L1 loss

            self.G_fake_loss_id = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_id_batch, logits=self.D_G_logits_id))

            self.G_fake_loss_age = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_age_batch, logits=self.D_G_logits_age))

            #loss of generator
            self.loss_G = self.G_loss_L1 + self.G_fake_loss_id + self.G_fake_loss_age + self.tv_loss

        with tf.name_scope('discriminator_loss'):
            #real image
            self.D_real_loss_id = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_id_batch, logits=self.D_input_logits_id))

            self.D_real_loss_age = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_age_batch, logits=self.D_input_logits_age))

            #fake image
            label_id_fake = tf.ones_like(self.label_id_batch) * self.num_person
            self.D_fake_loss_id = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label_id_fake, logits=self.D_G_logits_id))
            
            #loss of discriminator
            self.loss_D = self.D_real_loss_age + self.D_real_loss_id + self.D_fake_loss_id
            #self.loss_D = self.D_real_loss_age
        
        #*************************************trainable variables****************************************
        trainable_variables = tf.trainable_variables()
        print('trainable_variables', len(trainable_variables))

        #variables of encoder
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        #variables of decoder
        self.De_variables = [var for var in trainable_variables if 'De_' in var.name]
        #variables of discriminator
        self.D_variables = [var for var in trainable_variables if 'D_' in var.name]
        print(len(self.E_variables), len(self.De_variables), len(self.D_variables))


        #*************************************collect the summary**************************************
        self.logits_age_summary = tf.summary.histogram('logits_age', self.logits_age)
        self.logits_id_summary = tf.summary.histogram('logits_id', self.logits_id)
        self.logits_id_classify_summary = tf.summary.histogram('logits_id_classify', self.logits_id_classify)
        self.G_summary = tf.summary.image('generated_image', self.G)
        self.D_input_logits_id_summary = tf.summary.histogram('D_input_logits_id', self.D_input_logits_id)
        self.D_input_logits_age_summary = tf.summary.histogram('D_input_logits_age', self.D_input_logits_age)
        self.D_G_logits_id_summary = tf.summary.histogram('D_G_logits_id', self.D_G_logits_id)
        self.D_G_logits_age_summary = tf.summary.histogram('D_G_logits_age', self.D_G_logits_age)
        self.G_loss_L1_summary = tf.summary.scalar('G_loss_L1', self.G_loss_L1)
        self.E_loss_age_summary = tf.summary.scalar('E_loss_age', self.E_loss_age)
        self.E_loss_id_summary = tf.summary.scalar('E_loss_id', self.E_loss_id)
        self.D_real_loss_id_summary = tf.summary.scalar('D_real_loss_id', self.D_real_loss_id)
        self.D_real_loss_age_summary = tf.summary.scalar('D_real_loss_age', self.D_real_loss_age)
        self.D_fake_loss_id_summary = tf.summary.scalar('D_fake_loss_id', self.D_fake_loss_id)
        self.G_fake_loss_id_summary = tf.summary.scalar('G_fake_loss_id', self.G_fake_loss_id)
        self.G_fake_loss_age_summary = tf.summary.scalar('G_fake_loss_age', self.G_fake_loss_age)
        self.tv_loss_summary = tf.summary.scalar('tv_loss', self.tv_loss)
        self.loss_D_summary = tf.summary.scalar('loss_D', self.loss_D)
        self.loss_E_summary = tf.summary.scalar('loss_E', self.loss_E)
        self.loss_G_summary = tf.summary.scalar('loss_G', self.loss_G)

        #for saving graph and variables
        self.saver = tf.train.Saver(max_to_keep=0)


        """
        #************************************run time********************************************
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tf.global_variables_initializer().run()
        step = 0
        start = time.time()
        for i in range(1):
            rrow, rfname, rlabel_id, rlabel_age, rinput_batch, rlabel_id_fake = self.session.run([row, fname, label_id, label_age, self.input_batch, label_id_fake])
            #print(rrow)
            #print(rfname, rlabel_id, rlabel_age)
            #print(rlabel_id_fake)

            #summary = self.summary.eval()
            #step += 1
            #self.writer.add_summary(summary, step)
        print(time.time() - start)
        

        coord.request_stop()
        coord.join(threads)
        """

    def train(self,
              num_epochs = 100, #number of epoch
              learning_rate = 0.0002, #initial learning rate
              display_freq = 1000,
              summary_freq = 100,
              save_freq = 5000,
              ):
        #*********************************** optimizer *******************************************************
        global_step = tf.Variable(0, trainable=False, name='global_step')
        global_learning_rate = tf.train.exponential_decay(
            learning_rate = learning_rate, 
            global_step = global_step,
            decay_steps = self.data_size / self.batch_size * 2, #decay leanrning rate each 2 epochs
            decay_rate = 1.0, #learning rate decay (0, 1], 1 means no decay
            staircase = True,
        )
        #print(global_learning_rate.get_shape())

        beta1 = 0.5  # parameter for Adam optimizer
        with tf.name_scope('discriminator_train'):
            D_optimizer = tf.train.AdamOptimizer(
                learning_rate=global_learning_rate,
                beta1=beta1
            )
            
            D_grads_and_vars = D_optimizer.compute_gradients(self.loss_D, var_list=self.D_variables)
            D_train = D_optimizer.apply_gradients(D_grads_and_vars)
            """
            print('D_grads_and_vars', len(D_grads_and_vars))
            for var in self.D_variables:
                print(var.name)
            for grad, var in D_grads_and_vars:
                print(var.op.name)
                print(var)
                print(grad)
            exit(0)
            """
        
        with tf.name_scope('encoder_train'):
            with tf.control_dependencies([D_train]):
                E_optimizer = tf.train.AdamOptimizer(
                    learning_rate=global_learning_rate,
                    beta1=beta1
                )
                E_grads_and_vars = E_optimizer.compute_gradients(self.loss_E, var_list=self.E_variables)
                E_train = E_optimizer.apply_gradients(E_grads_and_vars)
        
        with tf.name_scope('generator_train'):
            with tf.control_dependencies([E_train]):
                G_optimizer = tf.train.AdamOptimizer(
                    learning_rate=global_learning_rate,
                    beta1=beta1
                )
                G_grads_and_vars = G_optimizer.compute_gradients(self.loss_G, var_list=self.E_variables + self.De_variables)
                G_train = G_optimizer.apply_gradients(G_grads_and_vars) #must be fetch
        
        #add movingaverage
        #ema = tf.train.ExponentialMovingAverage(0.99)
        #update_losses = ema.apply([self.loss_D, self.loss_E, self.loss_G]) #must be fetch
        incr_global_step = tf.assign(global_step, global_step+1) #must be fetch

        #*****************************************collect summary**********************************
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + '/values', var)
        
        for grad, var in D_grads_and_vars + E_grads_and_vars + G_grads_and_vars:
            #print(var.name)
            #print(grad)
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        
        with tf.name_scope('parameter_count'):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
        
        #***************************************tensorboard****************************************
        self.global_learning_rate_summary = tf.summary.scalar('global_learning_rate', global_learning_rate)
        self.summary = tf.summary.merge_all()
        #print('summary', self.summary)
        #self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'))
        
        #***************************************training*******************************************
        print('\n Preparing for training...')
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        #initialize the graph
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_batch_per_epoch = int(np.math.ceil(self.data_size / self.batch_size))
        #display_freq = num_batch_per_epoch
        max_steps = num_batch_per_epoch * num_epochs
        print('num_batch_per_epoch', num_batch_per_epoch)
        # epoch iteration
        for epoch in range(num_epochs):
            for batch_ind in range(num_batch_per_epoch):
                start_time = time.time()
                step = global_step.eval()
                def should(freq):
                    return (freq > 0) and ((step+1) % freq == 0 or step+1==max_steps)

                fetches = {
                    'train': G_train,
                    'incr_global_step': incr_global_step,
                    'D_err': self.loss_D,
                    'D_real_age_err': self.D_real_loss_age,
                    'D_real_id_err': self.D_real_loss_id,
                    'D_fake_id_err': self.D_fake_loss_id,
                    'E_err': self.loss_E,
                    'E_id_err': self.E_loss_id,
                    'E_age_err': self.E_loss_age,
                    'G_err': self.loss_G,
                    'G_L1_err': self.G_loss_L1,
                    'G_fake_id_err': self.G_fake_loss_id,
                    'G_fake_age_err': self.G_fake_loss_age,
                    'tv_err': self.tv_loss
                }

                if should(display_freq):
                    fetches['input_batch'] = self.input_batch
                    fetches['G'] = self.G

                if should(summary_freq):
                    fetches['summary'] = self.summary

                results = self.session.run(fetches)

                if should(display_freq):
                    print("saving display images")
                    name = '{:06d}_input.png'.format(global_step.eval())
                    self.save_image_batch(results['input_batch'], name)
                    name = '{:06d}_generated.png'.format(global_step.eval())
                    self.save_image_batch(results['G'], name)

                if should(summary_freq):
                    print("recording summary")
                    self.writer.add_summary(results['summary'], step+1)

                if should(save_freq):
                    print("saving model")
                    self.saver.save(self.session, os.path.join(checkpoint_dir, 'model'), global_step=global_step.eval())

                """
                _, _, D_err, D_real_age_err, D_real_id_err, D_fake_id_err, E_err, E_id_err, E_age_err, G_err, G_L1_err, \
                    G_fake_id_err, G_fake_age_err, tv_err = self.session.run(
                    fetches = [
                        G_train,
                        incr_global_step,
                        self.loss_D,
                        self.D_real_loss_age,
                        self.D_real_loss_id,
                        self.D_fake_loss_id,
                        self.loss_E,
                        self.E_loss_id,
                        self.E_loss_age,
                        self.loss_G,
                        self.G_loss_L1,
                        self.G_fake_loss_id,
                        self.G_fake_loss_age,
                        self.tv_loss
                    ]
                )
                """

                print('\nEpoch: [%d/%d] Batch: [%d/%d] G_err=%.4f E_err=%.4f D_err=%.4f' %
                    (epoch+1, num_epochs, batch_ind+1, num_batch_per_epoch, results['G_err'], results['E_err'], results['D_err']))
                print('\tG_L1_err=%.4f G_fake_id_err=%.4f G_fake_age_err=%.4f tv_err=%.4f' %
                    (results['G_L1_err'], results['G_fake_id_err'], results['G_fake_age_err'], results['tv_err']))
                print('\tE_id_err=%.4f E_age_err=%.4f' % (results['E_id_err'], results['E_age_err']))
                print('\tD_real_id_err=%.4f D_real_age_err=%.4f D_fake_id_err=%.4f' % (results['D_real_id_err'], results['D_real_age_err'], results['D_fake_id_err']))

                #estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batch_per_epoch + (num_batch_per_epoch - batch_ind - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                

        coord.request_stop()
        coord.join(threads)



    def encoder(self, image, reuse_variables=False):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        
        """
        layer_1: [batch, 128, 128, num_input_channels] => [batch, 64, 64, 64]
        layer_2: [batch, 64, 64, 64] => [batch, 32, 32, 128]
        layer_3: [batch, 32, 32, 128] => [batch, 16, 16, 256]
        layer_4: [batch, 16, 16, 256] => [batch, 8, 8, 512]
        layer_5: [batch, 8, 8, 512] => [batch, 4, 4, 1024]
        layer_6: [batch, 4, 4, 1024] => [batch, 2, 2, 2048]
        """
        num_layers = int(np.log2(self.image_size) - int(self.kernel_size/2)) #6
        print('num_layers', num_layers)
        layers = []

        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'E_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.kernel_size,
                    name=name
                )
            current = tf.nn.relu(current)

        # fully connection layer
        name = 'E_fc'
        current = fc(
            input_vector=tf.reshape(current, [self.batch_size, 2*2*2048]),
            num_output_length=self.num_z_channels,
            name=name
        )

        logits_age = tf.slice(current, [0, 0], [self.batch_size, 7])
        logits_id  = tf.slice(current, [0, 7], [self.batch_size, 320])

        name = 'E_fc_id_classify'
        logits_id_classify = fc(
            input_vector=logits_id,
            num_output_length=self.num_person,
            name=name
        )

        print(logits_age, logits_id, logits_id_classify)
        return logits_age, logits_id, logits_id_classify
    
    def decoder(self, z, labels_age, reuse_variables=False):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()

        """
        fc: [batch, 327] => [batch, 2, 2, 2048]
        #stride 2
        deconv_1: [batch, 2, 2, 2048] => [batch, 4, 4, 1024]
        deconv_2: [batch, 4, 4, 1024] => [batch, 8, 8, 512]
        deconv_3: [batch, 8, 8, 512] => [batch, 16, 16, 256]
        deconv_4: [batch, 16, 16, 256] => [batch, 32, 32, 128]
        deconv_5: [batch, 32, 32, 128] => [batch, 64, 64, 64]
        deconv_6: [batch, 64, 64, 64] => [batch, 128, 128, 32]

        #stride 1
        deconv_7: [batch, 128, 128, 32] => [batch, 128, 128, 16]
        deconv_8: [batch, 128, 128, 16] => [batch, 128, 128, 3]
        """
        num_layers = int(np.log2(self.image_size) - int(self.kernel_size/2)) #6

        labels_age_one_hot = tf.one_hot(labels_age, self.num_categories, on_value=1.0, off_value=-1.0, dtype=tf.float32)
        print('labels_age_one_hot', labels_age_one_hot)
        input = tf.concat([z, labels_age_one_hot], axis=1)
        print('input', input)
        mini_map_size = int(self.image_size / 2**num_layers) #2

        #fc layer
        name = 'De_fc'
        current = fc(
            input_vector=input,
            num_output_length=self.num_gen_channels * mini_map_size * mini_map_size,
            name=name
        )

        # reshape to cube for deconv
        current = tf.reshape(current, [-1, mini_map_size, mini_map_size, self.num_gen_channels])
        current = tf.nn.relu(current)

        # deconv layers with stride 2
        for i in range(num_layers):
            name = 'De_deconv' + str(i)
            current = deconv2d(
                    input_map=current,
                    output_shape=[self.batch_size,
                                  mini_map_size * 2 ** (i + 1),
                                  mini_map_size * 2 ** (i + 1),
                                  int(self.num_gen_channels / 2 ** (i + 1))],
                    size_kernel=self.kernel_size,
                    name=name
                )
            current = tf.nn.relu(current)
        #[batch, 128, 128, 32] => [batch, 128, 128, 16]
        name = 'De_deconv' + str(i+1)
        current = deconv2d(
                    input_map=current,
                    output_shape=[self.batch_size,
                                  self.image_size,
                                  self.image_size,
                                  int(self.num_gen_channels / 2 ** (i + 2))],
                    size_kernel=self.kernel_size,
                    stride=1,
                    name=name
                )
        current = tf.nn.relu(current)
        #[batch, 128, 128, 16] => [batch, 128, 128, 3]
        name = 'De_deconv' + str(i+2)
        current = deconv2d(
                    input_map=current,
                    output_shape=[self.batch_size,
                                  self.image_size,
                                  self.image_size,
                                  self.num_input_channels],
                    size_kernel=self.kernel_size,
                    stride=1,
                    name=name
                )
        
        return tf.nn.tanh(current)

    def discriminator(self, image, is_training=True, reuse_variables=False, enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()

        """
        layer_1: [batch, 128, 128, num_input_channels] => [batch, 64, 64, 64]
        layer_2: [batch, 64, 64, 64] => [batch, 32, 32, 128]
        layer_3: [batch, 32, 32, 128] => [batch, 16, 16, 256]
        layer_4: [batch, 16, 16, 256] => [batch, 8, 8, 512]
        layer_5: [batch, 8, 8, 512] => [batch, 4, 4, 1024]
        layer_6: [batch, 4, 4, 1024] => [batch, 2, 2, 2048]
        """
        num_layers = int(np.log2(self.image_size) - int(self.kernel_size/2)) #6
        print('num_layers', num_layers)

        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'D_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.kernel_size,
                    name=name
                )
            if enable_bn:
                name = 'D_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = tf.nn.relu(current)

        # fully connection layer
        name = 'D_fc'
        current = fc(
            input_vector=tf.reshape(current, [self.batch_size, 2*2*2048]),
            num_output_length=1024,
            name=name
        )
        current = lrelu(current)

        name = 'D_logits_age'
        logits_age = fc(
            input_vector=current,
            num_output_length=self.num_categories,
            name=name
        )

        name = 'D_logits_id'
        logits_id = fc(
            input_vector=current,
            num_output_length=self.num_person+1,
            name=name
        )
        return logits_id, logits_age

    
    
    def save_image_batch(self, image_batch, name):
        sample_dir = os.path.join(self.save_dir, 'sample')
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        #transform the pixel value from -1~1 to 0~1
        images = (image_batch + 1) / 2.0
        frame_size = int(np.sqrt(self.batch_size))
        img_h, img_w = image_batch.shape[1], image_batch.shape[2]
        frame = np.zeros([img_h * frame_size, img_w * frame_size, 3])

        for ind, image in enumerate(images):
            ind_row = ind % frame_size
            ind_col = ind // frame_size
            frame[(img_h*ind_row):(img_h*ind_row+img_h), (img_w*ind_col):(img_w*ind_col+img_w), :] = image

        imsave(os.path.join(sample_dir, name), frame)

    def test(self):
        pass




        


        


        

