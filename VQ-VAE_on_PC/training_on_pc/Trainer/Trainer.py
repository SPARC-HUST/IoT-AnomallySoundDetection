from tensorflow.keras import models, metrics
from tensorflow import GradientTape, reduce_mean, transpose, sigmoid, ones
from Models import get_model


class ModelTrainer(models.Model):
    def __init__(self, name, train_variance, og_dim,
                 transfer_learning, **kwargs):    # noqa: E501
        super(ModelTrainer, self).__init__()
        self.train_variance = train_variance

        self.model = get_model(name, og_dim)

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = metrics.Mean(name="vq_loss")
        self.supervised_loss_tracker = metrics.Mean(name="supervised_loss")

        # setting up parameters for transfer learning
        self.flag = transfer_learning
        if self.flag:
            self.beta = kwargs['beta']
            self.training_anomaly = kwargs['training_anomaly']
            self.BATCH_SIZE = kwargs['batch_size']
            self.ANOM_SIZE = self.training_anomaly.shape[0]
            self.based_model_path = kwargs['based_model']
            # load the pretrained weights if transfer learning mode is on
            based_model = models.load_model(self.based_model_path)
            # call the model once to initialize the weights
            self.model(ones(self.training_anomaly.shape))
            self.model.set_weights(based_model.get_weights())

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
            self.supervised_loss_tracker
        ]

    def train_step(self, x):
        # training
        with GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.model(x)

            # Calculate the losses.
            normal_loss = (
                reduce_mean((x - reconstructions) ** 2, axis=(1, 2, 3), keepdims=True) / self.train_variance    # noqa: E501
            )

            reconstruction_loss = reduce_mean(normal_loss)

            # transfer learning loss
            supervised_loss = 0
            if self.flag:
                abnorm_recon = self.model(self.training_anomaly)
                anom_loss = transpose(
                        reduce_mean((self.training_anomaly - abnorm_recon) ** 2, axis=(1, 2, 3), keepdims=True) / self.train_variance    # noqa: E501
                )
                supervised_loss = self.beta*reduce_mean(
                        sigmoid(anom_loss - normal_loss)
                )

            total_loss = reconstruction_loss + sum(self.model.losses) - supervised_loss    # noqa: E501

        # Backpropagation.
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))    # noqa: E501

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.model.losses))
        self.supervised_loss_tracker.update_state(supervised_loss)

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
            "supervised_loss": self.supervised_loss_tracker.result()
        }
