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

import json
import os
import nibabel as nib

import torch
import torch.optim as optim
from learners.supervised_learner import SupervisedLearner
from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets.unet import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)

from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)

#from utils.custom_client_datalist_json_path import custom_client_datalist_json_path

from learners.simple_network import SimpleNetwork

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss
import monai
from monai.inferers import sliding_window_inference

##Iman Added for the json files
import os
import glob
import json

def load_pet(filename):
    with open(filename) as f:
        pet = json.loads(f.read())
    return pet



class SupervisedMonaiProstateLearner(SupervisedLearner):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        """MONAI Learner for prostate segmentation task.
        It inherits from SupervisedLearner.

        Args:
            train_config_filename: path for config file, this is an addition term for config loading
            aggregation_epochs: the number of training epochs for a round.
            train_task_name: name of the task to train the model.

        Returns:
            a Shareable with the updated local model after running `execute()`
        """
        super().__init__(
            aggregation_epochs=aggregation_epochs,
            train_task_name=train_task_name,
        )
        self.train_config_filename = train_config_filename
        self.config_info = None

    def train_config(self, fl_ctx: FLContext):
        """MONAI traning configuration
        Here, we use a json to specify the needed parameters
        """

        # Load training configurations json
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_job_id())
        train_config_file_path = os.path.join(app_config_dir, self.train_config_filename)
        if not os.path.isfile(train_config_file_path):
            self.log_error(
                fl_ctx,
                f"Training configuration file does not exist at {train_config_file_path}",
            )
        with open(train_config_file_path) as file:
            self.config_info = json.load(file)

        print(f'\n\n\n\n\n\nconfig_info={self.config_info}\n\n\n\n\n\n\n\n\n\n')

        # Get the config_info
        self.lr = self.config_info["learning_rate"]

        #Iman Added
        self.val_freq = self.config_info["validation_freq"]
        self.best_acc = 0.0
        self.batch_of_train = self.config_info["batch_of_train"]
        self.batch_of_val = self.config_info["batch_of_val"]


        self.fedproxloss_mu = self.config_info["fedproxloss_mu"]
        cache_rate = self.config_info["cache_dataset"]
        #dataset_base_dir = self.config_info["dataset_base_dir"]
        #datalist_json_path = self.config_info["datalist_json_path"]
        self.roi_size = self.config_info.get("roi_size", (224, 224, 32))
        self.infer_roi_size = self.config_info.get("infer_roi_size", (224, 224, 32))

        # Get datalist json
        #datalist_json_path = custom_client_datalist_json_path(datalist_json_path, self.client_id)

        # Set datalist
        # Set datalist

        data_list_file_path = os.path.join(self.app_root, self.data_list_json_file)
        base_dir = os.path.join(self.dataset_root, self.dataset_folder_name)








        ##Iman Added
        self.loaded_dic_files = load_pet(data_list_file_path)
        self.input_file_example_path = self.loaded_dic_files["training"][0]['image']
        self.val_file_example_path = self.loaded_dic_files["training"][0]['label']

        example_path = os.path.join(base_dir,self.input_file_example_path)
        print(f'\n\none example of path:{example_path}')

        self.x_file = nib.load(os.path.join(base_dir,self.input_file_example_path))
        self.y_file = nib.load(os.path.join(base_dir,self.val_file_example_path))


        train_list = load_decathlon_datalist(
            data_list_file_path=data_list_file_path,
            is_segmentation=True,
            data_list_key="training",
            base_dir=base_dir,
        )
        valid_list = load_decathlon_datalist(
            data_list_file_path=data_list_file_path,
            is_segmentation=True,
            data_list_key="validation",
            base_dir=base_dir,
        )

        self.log_info(
            fl_ctx,
            f"Training Size: {len(train_list)}, Validation Size: {len(valid_list)}",
        )

        # Set the training-related context
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNetwork().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)


        ## Iman Added for the image synthesis
        self.criterion = torch.nn.L1Loss()

        if self.fedproxloss_mu > 0:
            self.log_info(fl_ctx, f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedproxloss_mu)



        D = 192
        H = 192
        W = 192
        self.transform_train = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys=["image"]),
                ScaleIntensityRanged(keys=["label"], a_min=0.0, a_max=2000.0, b_min=0.0, b_max=1.0),
                RandSpatialCropd(keys=["image", "label"], roi_size=[D, H, W], random_center=True, random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                #RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                #RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        self.transform_valid = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys=["image"]),
                ScaleIntensityRanged(keys=["label"], a_min=0.0, a_max=2000.0, b_min=0.0, b_max=1.0),
            ]
        )





        #self.transform_post = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # Set dataset
        if cache_rate > 0.0:
            self.train_dataset = CacheDataset(
                data=train_list,
                transform=self.transform_train,
                cache_rate=cache_rate,
                num_workers=1,
            )
            self.valid_dataset = CacheDataset(
                data=valid_list,
                transform=self.transform_valid,
                cache_rate=cache_rate,
                num_workers=1,
            )
        else:
            self.train_dataset = Dataset(
                data=train_list,
                transform=self.transform_train,
            )
            self.valid_dataset = Dataset(
                data=valid_list,
                transform=self.transform_valid,
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_of_train,
            shuffle=True,
            num_workers=1,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_of_val,
            shuffle=False,
            num_workers=1,
        )
        print(f'\n\n self.infer_roi_size={self.infer_roi_size}')
        self.valid_metric = self.criterion


    # define inference method
    def inferer(self, input, model):

            def _compute(input, model):
                return sliding_window_inference(
                    inputs=input,
                    roi_size=self.infer_roi_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                )

            return _compute(input, model)

        #self.inferer = SlidingWindowInferer(roi_size=self.infer_roi_size, sw_batch_size=4, overlap=0.25)
