#!/usr/bin/env/python3
import os
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from dataio.data_prepare_singleprocess import dataio_prep, data_prepare, check_folders

# Brain class for speech enhancement training
class GTCRNBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
        """Apply masking to convert from noisy waveforms to enhanced signals.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            A dictionary with keys {"spec", "wav"} with predicted features.
        """
        # We first move the batch to the appropriate device, and
        # compute the features necessary for masking.
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.noisy_sig  # (B, T)
        clean_wavs, _ = batch.clean_sig
        '''
        noisy_spec = batch.spec
        self.clean_wavs, self.lens = batch.clean_sig
        noisy_wavs, self.lens = self.hparams.wav_augment(
            self.clean_wavs, self.lens
        )
        '''
        enhanced_wavs = self.modules.model(noisy_wavs)
        '''
        # Also return predicted wav, for evaluation. Note that this could
        # also be used for a time-domain loss term.
        predict_wav = self.hparams.resynth(
            torch.expm1(predict_spec), noisy_wavs
        )
        '''

        # Return a dictionary so we don't have to remember the order
        return {
            "wav": enhanced_wavs,
            "clean_wavs": clean_wavs,
            "lens": lens
        }

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`. (predicted spectrogram)
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        # Prepare clean targets for comparison
        clean_wavs, lens = batch.clean_sig

        # Directly compare the masked spectrograms with the clean targets
        loss = sb.nnet.losses.mse_loss(
            predictions["wav"], clean_wavs, lens
        )

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id,
            predictions["wav"],
            clean_wavs,
            lens,
            reduction="batch",
        )

        # Some evaluations are slower, and we only want to perform them
        # on the validation set.
        if stage != sb.Stage.TRAIN:

            # Evaluate speech intelligibility as an additional metric
            self.stoi_metric.append(
                batch.id,
                predictions["wav"],
                clean_wavs,
                lens,
                reduction="batch",
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.mse_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.stoi_metric = sb.utils.metric_stats.MetricStats(
                metric=sb.nnet.loss.stoi_loss.stoi_loss
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "stoi": -self.stoi_metric.summarize("average"),
            }

        # At the end of validation, we can write stats and checkpoints
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best STOI score.
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["stoi"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


# Recipe begins!
if __name__ == "__main__":
    curr_path = os.path.abspath(os.path.dirname(__file__))
    project_name = Path(curr_path).name
    arg_list = [
        f'{curr_path}/train.yaml',
        '--curr_path', f'{curr_path}',
        '--project_name', f'{project_name}',
        '--device', 'cuda:0',
    ]

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(arg_list)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    if not check_folders(hparams["data_folder"]):
        sb.utils.distributed.run_on_main(
            data_prepare,
            kwargs=hparams,
        )
    #sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])

    # Create dataset objects "train" and "valid"
    datasets = dataio_prep(**hparams)

    # Initialize the Brain object to prepare for mask training.
    se_brain = GTCRNBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load best checkpoint (highest STOI) for evaluation
    test_stats = se_brain.evaluate(
        test_set=datasets["test"],
        max_key="stoi",
        test_loader_kwargs=hparams["dataloader_options"],
    )
