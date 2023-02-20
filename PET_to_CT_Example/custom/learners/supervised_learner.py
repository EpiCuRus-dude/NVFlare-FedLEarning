# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import nibabel as nib
import copy
from abc import abstractmethod
import os
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ValidateType



class SupervisedLearner(Learner):
    def __init__(
        self,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        """Simple Supervised Trainer.
            This provides the basic functionality of a local learner: perform before-train validation on
            global model at the beginning of each round, perform local training, and send the updated weights.
            No model will be saved locally, tensorboard record for local loss and global model validation score.
            Enabled both FedAvg and FedProx

        Args:
            train_config_filename: directory of config file.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            train_task_name: name of the task to train the model.

        Returns:
            a Shareable with the updated local model after running `execute()`
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.aggregation_epochs = aggregation_epochs
        self.train_task_name = train_task_name
        self.best_metric = 0.0
        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0
        # FedProx related
        self.fedproxloss_mu = 0.0
        self.criterion_prox = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # when a run starts, this is where the actual settings get initialized for trainer

        # set the paths according to fl_ctx
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = ws.get_app_dir(fl_ctx.get_job_id())

        # get and print the args
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with args: \n {fl_args}",
        )

        # set local tensorboard writer for local validation score of global model
        self.writer = SummaryWriter(app_dir)
        # set the training-related contexts, this is task-specific




        ### Iman ADDED
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        self.dataset_root = self.workspace.get_root_dir()
        self.fl_args = fl_ctx.get_prop(FLContextKey.ARGS)

        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")



        self.wf_config_file_name = self.fl_args.train_config

        with open(os.path.join(self.app_root, self.wf_config_file_name)) as file:
            wf_config = json.load(file)

        self.wf_config = wf_config

        self.data_list_json_file = wf_config["data_list_json_file"]
        self.dataset_folder_name = wf_config["dataset_folder_name"]


        self.path_train = os.path.join(self.app_root,"train_samp")
        if not os.path.exists(self.path_train):
            os.mkdir(self.path_train)


        self.path_val = os.path.join(self.app_root,"val_samp")
        if not os.path.exists(self.path_val):
            os.mkdir(self.path_val)



        print(f'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\\n\n\n\n\n\n\n\n\n\n\n******************************************************************\n******************************************************\n')
        print(f'app_root={self.app_root}\n')
        print(f'self.data_list_json_file={self.data_list_json_file}\n')

        print(f'workspace={self.workspace}\n')
        print(f'dataset_root={self.dataset_root}\n')
        print(f'fl_args={self.fl_args}\n')
        print(f'*************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************\n\n\n\n\n\n\n\n\n\n\n\n\n\n')


        ### Iman: it should be here
        self.train_config(fl_ctx)




    @abstractmethod
    def train_config(self, fl_ctx: FLContext):
        """Traning configurations customized to individual tasks
        This can be specified / loaded in any ways
        as long as they are made available for further training and validation
        some potential items include but not limited to:
        self.lr
        self.fedproxloss_mu
        self.model
        self.device
        self.optimizer
        self.criterion
        self.transform_train
        self.transform_valid
        self.transform_post
        self.train_loader
        self.valid_loader
        self.inferer
        self.valid_metric
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self, fl_ctx: FLContext):
        # collect threads, close files here
        pass

    def local_train(
        self,
        fl_ctx,
        train_loader,
        model_global,
        abort_signal: Signal,
    ):
        """Typical training logic
        Total local epochs: self.aggregation_epochs
        Load data pairs from train_loader: image / label
        Compute outputs with self.model
        Compute loss with self.criterion
        Add fedprox loss
        Update model
        """

        print(f'\n\n\n\n\n**************** **************** **************** **************** Local Training Started:')
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.model.train()

            epoch_len = len(train_loader)

            self.epoch_global = self.epoch_of_start_time + epoch

            self.log_info(
                fl_ctx,
                f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})",
            )

            #Iman Added
            avg_loss = 0.0
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)


                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model, model_global)
                    loss += fed_prox_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                current_step = epoch_len * self.epoch_global + i

                avg_loss += loss.item()



                self.writer.add_scalar("train_loss", loss.item(), current_step)


                ##Iman Added to save the images
                if i%50==0:
                    self.save_samples(i, inputs, labels, outputs, "train")


            #Iman Changed
            self.log_info(
                fl_ctx,
                f"train_loss {self.client_id}: avg_loss:{avg_loss} (epoch={epoch+1}, inputs_shape={inputs.shape}, labels_shape={labels.shape}, outputs_shape={outputs.shape})\n\n\n",
            )
            self.writer.add_scalar("train_loss_epoch", avg_loss , epoch)



            #Iman Added for the validation inside the locaal training
            if self.val_freq > 0 and epoch % self.val_freq == 0:
                acc = self.local_valid(
                    self.model,
                    self.valid_loader,
                    abort_signal,
                    tb_id="val_metric_global_model",
                    record_epoch=self.epoch_global,
                )
                self.log_info(
                    fl_ctx,
                    f"Accuracy of the validation {acc}, (epoch={epoch+1})\n\n\n",
                )
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_model(is_best=True)
                    self.log_info(
                        fl_ctx,
                        f"Model is saved for this epoch, (epoch={epoch+1} and the acuuracy is improved from {self.best_acc} to {acc} )\n\n\n\n",
                    )
        print(f'\n\n\n**************** **************** **************** **************** Local Training Done\n\n\n')

    ##IMan Added from Cifar10 Example
    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights, "epoch": self.epoch_global}
        if is_best:
            save_dict.update({"best_acc": self.best_acc})
            torch.save(save_dict, self.best_local_model_file)
        else:
            torch.save(save_dict, self.local_model_file)

    def save_samples(self, idx, inputs, labels, outputs, type=None):
            ##############################################################

            #print(np.shape(outputs))

            if type=="train":
                path = self.path_train
            else:
                path = self.path_val

            predW = outputs.cpu().detach().numpy()[0,0,:,:,:]
            #print(f'np.shape(predW)={np.shape(predW)}')
            pred_img = nib.Nifti1Image(predW, self.y_file.affine, self.y_file.header)
            path_img = path + '/'+str(idx) +'_Pred.nii'
            #print(f'path_img={path_img}')
            nib.save(pred_img, path_img)

            x = inputs.cpu().detach().numpy()[0,0,:,:,:]
            x_img = nib.Nifti1Image(x, self.x_file.affine, self.x_file.header)
            path_img_x = path + '/'+str(idx) +'_X.nii'
            #print(f'path_img={path_img_x}')
            nib.save(x_img, path_img_x)

            y = labels.cpu().detach().numpy()[0,0,:,:,:]
            #print(f'np.shape(y)={np.shape(y)}')
            y_img = nib.Nifti1Image(y, self.y_file.affine, self.y_file.header)
            path_img_y = path + '/'+str(idx) +'_Y.nii'
                #print(f'path_img={path_img_y}')
            nib.save(y_img, path_img_y)
            ############################################################




    def local_valid(
        self,
        model,
        valid_loader,
        abort_signal: Signal,
        tb_id=None,
        record_epoch=None,
    ):
        """Typical validation logic
        Load data pairs from train_loader: image / label
        Compute outputs with self.model
        Perform post transform (binarization, etc.)
        Compute evaluation metric with self.valid_metric
        Add score to tensorboard record with specified id
        """
        model.eval()
        with torch.no_grad():
            metric = 0
            for i, batch_data in enumerate(valid_loader):
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                val_images = batch_data["image"].to(self.device)
                val_labels = batch_data["label"].to(self.device)


                print(f'\n\n ************************************inside the validation,\n val_images={val_images.shape}, val_labels={val_labels.shape}')

                #val_images = val_images.unsqueeze(1)

                #print(f'\nval_images={val_images.shape}')

                # Inference
                val_outputs = self.inferer(val_images, model)

                print(f'\nval_outputs={val_outputs.shape}')

                #val_outputs = self.transform_post(val_outputs)
                # Compute metric
                metric_score = self.valid_metric(val_outputs, val_labels)

                #Iman Added for image synthesis, change it for the segmentation tasks
                metric_score = 1 - metric_score

                metric += metric_score.item()


                if i%10==0:
                    self.save_samples( i, val_images, val_labels, val_outputs, "val")
            # compute mean dice over whole validation set
            metric /= len(valid_loader)


            # tensorboard record id, add to record if provided
            if tb_id:
                self.writer.add_scalar(tb_id, metric, record_epoch)
        return metric

    def train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """Typical training task pipeline with potential HE and fedprox functionalities
        Get global model weights (potentially with HE)
        Prepare for fedprox loss
        Local training
        Return updated weights (model_diff)
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")


        #print(f'\n\n\n\n\n\n\n self.loaded_dic_files["training"]={self.loaded_dic_files["training"]}\n\n\n\n')

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model.load_state_dict(local_var_dict)

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference for potential FedProx loss
        if self.fedproxloss_mu > 0:
            model_global = copy.deepcopy(self.model)
            for param in model_global.parameters():
                param.requires_grad = False
        else:
            model_global = None

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            model_global=model_global,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        for name in global_weights:
            if name not in local_weights:
                continue
            model_diff[name] = local_weights[name].cpu().numpy() - global_weights[name]
            if np.any(np.isnan(model_diff[name])):
                self.system_panic(f"{name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # flush the tb writer
        self.writer.flush()

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        return dxo.to_shareable()

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Typical validation task pipeline with potential HE functionality
        Get global model weights (potentially with HE)
        Validation on local data
        Return validation score
        """
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # validation on global model
        model_owner = "global_model"

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for validation! Received weight dict is {global_weights}")

        # before_train_validate only, can extend to other validate types
        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_metric = self.local_valid(
                self.model,
                self.valid_loader,
                abort_signal,
                tb_id="val_metric_global_model",
                record_epoch=self.epoch_global,
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"val_metric_global_model ({model_owner}): {global_metric:.4f}")
            # validation metrics will be averaged with weights at server end for best model record
            metric_dxo = DXO(
                data_kind=DataKind.METRICS,
                data={MetaKey.INITIAL_METRICS: global_metric},
                meta={},
            )
            metric_dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, len(self.valid_loader))
            return metric_dxo.to_shareable()
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)
