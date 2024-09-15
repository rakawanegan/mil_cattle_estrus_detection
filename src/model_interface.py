import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl


class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, net,):
        super(ModelInterface, self).__init__()
        self.net = net
        self.loss = nn.BCELoss()
        self.n_classes = self.net.n_classes

        #---->Metrics
        if self.n_classes > 2:
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection(
                [
                    torchmetrics.Accuracy(num_classes=self.n_classes, average='micro', threshold=self.net.threshold),
                    torchmetrics.CohenKappa(num_classes=self.n_classes, threshold=self.net.threshold),
                    torchmetrics.F1Score(num_classes=self.n_classes, average = 'macro', threshold=self.net.threshold),
                    torchmetrics.Recall(average='macro', num_classes = self.n_classes, threshold=self.net.threshold),
                    torchmetrics.Precision(average='macro', num_classes = self.n_classes, threshold=self.net.threshold),
                    torchmetrics.Specificity(average='macro', num_classes = self.n_classes, threshold=self.net.threshold),
                ]
            )
        else :
            self.AUROC = torchmetrics.AUROC(num_classes=2, average = 'macro', task='binary')
            metrics = torchmetrics.MetricCollection(
                [
                    torchmetrics.Accuracy(num_classes=self.n_classes, average='macro', task='binary', threshold=self.net.threshold),
                    torchmetrics.F1Score(num_classes=self.n_classes, average='macro', task='binary', threshold=self.net.threshold),
                    torchmetrics.Recall(average='macro', num_classes=self.n_classes, task='binary', threshold=self.net.threshold),
                    torchmetrics.Precision(average='macro', num_classes=self.n_classes, task='binary', threshold=self.net.threshold),
                    torchmetrics.Specificity(average='macro', num_classes=self.n_classes, task='binary', threshold=self.net.threshold),
                ]
            )
        self.valid_metrics = metrics.clone(prefix = 'val_')

        self.output_on_validation = list()

    def forward(self, x):
        return self.net(x)

    def get_embedding(self, x):
        return self.net.get_embedding(x)

    def training_step(self, batch, batch_idx):
        #---->inference
        x, t = batch
        results_dict = self.net(x)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->loss
        loss = self.loss(Y_prob, t)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, t = batch
        results_dict = self.net(x)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        self.output_on_validation.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : t})

    def on_validation_epoch_end(self):
        outputs = self.output_on_validation  # Collect outputs from all devices/GPUs

        logits = torch.cat([out['logits'] for out in outputs], dim=0)
        Y_prob = torch.cat([out['Y_prob'] for out in outputs], dim=0)
        Y_hat = torch.cat([out['Y_hat'] for out in outputs], dim=0)
        label = torch.cat([out['label'] for out in outputs], dim=0)

        # Compute metrics based on concatenated outputs
        self.valid_metrics(Y_prob, label)
        # Log metrics
        self.log_dict(
            {name: metric for name, metric in self.valid_metrics.compute().items()},
            on_epoch=True, prog_bar=True
        )
        # self.log('val_max_prob', torch.max(Y_prob), on_epoch=True, prog_bar=True)
        # self.log('val_num_detected', int(torch.sum(Y_hat)), on_epoch=True, prog_bar=True)

        # Reset output_on_validation
        self.output_on_validation = list()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        return [optimizer]
