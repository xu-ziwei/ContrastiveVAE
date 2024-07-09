import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Flatten, Dense, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras import backend as K
from tqdm import tqdm
import datetime

class LVAE(tf.keras.Model):
    def __init__(self, input_dims, latent_dim=32, num_classes=4, beta=1.0,gamma=0.001, optimizer=None, **kwargs):
        super(LVAE, self).__init__(**kwargs)
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma

        # Build the encoder, decoder, and classifier
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.classifier = self.build_classifier()

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        else:
            self.optimizer = optimizer

    def build_encoder(self):
        inputs = Input(shape=self.input_dims)
        x = Conv3D(32, kernel_size=3, activation='relu', strides=2, padding='same')(inputs)
        x = Conv3D(64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        return Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,))

        # Calculate the dimensions before flattening
        dim1 = self.input_dims[0] // 4
        dim2 = self.input_dims[1] // 4
        dim3 = self.input_dims[2] // 4
        channels = 64
        dense_units = dim1 * dim2 * dim3 * channels

        x = Dense(dense_units, activation='relu')(latent_inputs)
        x = Reshape((dim1, dim2, dim3, channels))(x)
        x = Conv3DTranspose(64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
        x = Conv3DTranspose(32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
        outputs = Conv3DTranspose(1, kernel_size=3, activation='sigmoid', padding='same')(x)
        
        return Model(latent_inputs, outputs, name='decoder')

    def build_classifier(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(128, activation='relu')(latent_inputs)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        return Model(latent_inputs, outputs, name='classifier')

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def loss_function(self, recon_x, x, logits, labels, mu, logvar):
        # Reconstruction loss
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        recon_x_flat = K.flatten(recon_x)
        x_flat = K.flatten(x)
        BCE = tf.reduce_sum(binary_crossentropy(x_flat, recon_x_flat))/batch_size

        # KL divergence loss
        KLD = (-0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar)))/batch_size

        # Determine classification loss weight
        # if label is [1,0,0,0] set w_l to 1
        w_l = tf.cast(tf.not_equal(labels[:, 0], 1), tf.float32)
        # w_l = 0
        
        # Classification loss
        CE = categorical_crossentropy(labels, logits)
        CE = w_l * CE  # Apply the weight
        CE = tf.reduce_sum(CE)/batch_size

        total_loss = BCE + self.beta*KLD + self.gamma*CE

        return total_loss, BCE, self.beta*KLD, self.gamma*CE

    def call(self, inputs):
        images, labels = inputs
        z_mean, z_log_var, z = self.encoder(images)
        reconstructions = self.decoder(z)
        logits = self.classifier(z)
        return reconstructions, logits, z_mean, z_log_var

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            reconstructions, logits, z_mean, z_log_var = self((images, labels), training=True)
            total_loss, BCE, KLD, CE = self.loss_function(recon_x=reconstructions, x=images, logits=logits, labels=labels, mu=z_mean, logvar=z_log_var)
                
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Log losses for TensorBoard
        self.compiled_metrics.update_state(labels, logits)
        return {"loss": total_loss, "BCE": BCE, "KLD": KLD, "CE": CE}

    def test_step(self, data):
        images, labels = data
        reconstructions, logits, z_mean, z_log_var = self((images, labels), training=False)
        total_loss, BCE, KLD, CE = self.loss_function(recon_x=reconstructions, x=images, logits=logits, labels=labels, mu=z_mean, logvar=z_log_var)

        # Log losses for TensorBoard
        self.compiled_metrics.update_state(labels, logits)
        return {"loss": total_loss, "BCE": BCE, "KLD": KLD, "CE": CE}

    def model_summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.classifier.summary()

