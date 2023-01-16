import tensorflow as tf
from core.Trainer import ModelTrainer

class TL_Trainer(ModelTrainer):
    def __init__(self, cfg, from_config=True, **kwargs):
        super().__init__(cfg)
        self.beta = cfg.TRANSFER_LEARNING.BETA if from_config else kwargs['beta']
        self.training_anomaly = kwargs['training_anomaly']
        self.batch_size = cfg.TRANSFER_LEARNING.ANOM_BATCH_SIZE \
            if from_config else kwargs['batch_size']
        self.based_model_path = cfg.TRANSFER_LEARNING.BASED_WEIGHTS \
            if from_config else kwargs['based_model']
        # load the pretrained weights if transfer learning mode is on
        self.load_pretrained_weights(self.based_model_path)
        # tf.concat(temp, axis=0)

    # override metrics
    def _setup_metrics(self):
        self.trackers = {
            'total_loss': tf.keras.metrics.Mean(name="total_loss"),
            'reconstruction_loss': tf.keras.metrics.Mean(name="reconstruction_loss"),
            'model_loss': tf.keras.metrics.Mean(name="model_loss"),
            'supervised_loss': tf.keras.metrics.Mean(name="supervised_loss")
        }

    # compute loss in transfer learning
    # @tf.function
    def _compute_loss(self, original, reconstruction):
        print("----------compute loss 1-------")
        sample_wise_loss = self._reconstruction_loss_sample_wise(original, reconstruction)
        print("----------sample_wise_loss 1-------")
        anom_loss = self._anomaly_loss()
        print("----------anom_loss 1-------")
        supervised_loss = self.beta*tf.reduce_mean(
                        tf.sigmoid(anom_loss - sample_wise_loss)
                )
        return {'reconstruction_loss': tf.reduce_mean(sample_wise_loss), \
                'supervised_loss': supervised_loss}, tf.squeeze(sample_wise_loss)

    def _anomaly_loss(self):
        temp = []
        for features, _, _ in self.training_anomaly:
            processed_feature = self.pre_prc.add_dimentsion(self.pre_prc.rescale(features))
            anom_recon = self.model(processed_feature)
            temp.append(tf.reduce_mean((processed_feature - anom_recon)**2, axis=(1, 2, 3), keepdims=True))
        return tf.transpose(tf.concat(temp, axis=0))

    def fit(self, data_dict):
        print(f"Normal_tl {self._get_number_of_samples(data_dict['train'])}")
        print(f"Anomaly tl {self._get_number_of_samples(self.training_anomaly)}")
        
        super().fit(data_dict)