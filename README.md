 [Documentation](https://docs.nerf.studio/)
 

# Images or Video
https://docs.nerf.studio/quickstart/custom_dataset.html#images-or-video
## Processing Data
`ns-process-data {images, video} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}`

ns-process-data video --help:

 * **--matching-method** {exhaustive,sequential,vocab_tree}  (default: sequential)   https://radiancefields.com/the-definitive-nerfstudio-command-guide
 * **--num-downscales** INT  Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the images by 2x, 4x, and 8x. (default: 3)
 * **--num-frames-target** INT Target number of frames to use per video, results may not be exact. (default: 300)，慎用，600张很慢       



## Training on your data
`ns-train nerfacto --data {PROCESSED_DATA_DIR}`

`ns-train nerfacto --data {PROCESSED_DATA_DIR} --steps-per-save 2000000  --max-num-iterations 4000000`

## viewer

Given a pretrained model checkpoint, you can start the viewer by running

```bash
ns-viewer --load-config {outputs/.../config.yml}
```

viewer的使用

render中有关键帧。可以添加多个时间作为关键帧。点击下面的play，就可以看见场景中有个绿色的在动。点击preview render就可以亲临实地观看。

在上面的connection diagnostics中可以选择是否显示fps也就是WebGL Statistics。下面的眼睛点亮点灭还能选择是否显示各种各样的东西。

## eval

`ns-eval --load-config --output-path name.json --render-output-path eval_render`

# Quickstart

The quickstart will help you get started with the default vanilla NeRF trained on the classic Blender Lego scene.
For more complex changes (e.g., running with your own data/setting up a new NeRF graph), please refer to our [references](#learn-more).

# downsample

https://github.com/nerfstudio-project/nerfstudio/issues/3221
ns-train splatfacto nerfstudio-data --data my data\colmappano_8  --downscale-factor 1

## 1. Installation: Setup the environment

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name nerfstudio  python=3.8
conda activate nerfstudio
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.7 and CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

需要安装cuda11.7参考[Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)

如果tiny cudann报错：https://github.com/NVlabs/tiny-cuda-nn/issues/385

连不上github可以先pip install ninja 然后通过git命令把tiny cuda nn库下载下来，然后进入torch下pip install . 因为可能要求gcc -v是9.4 9.5但机器只有7.多

如果环境配好了但是splatnerf出了问题是因为ninja，那么可以读一读报错，把某个h头文件复制到某个位置就行。

注意系统cuda版本

如果报ld:cannot find -lcuda:No such file or directory，那么export LIBRARY_PATH=/usr/local/cuda-11.8/lib64/stubs:$LIBRARY_PATH

如果报cannot import name 'csrc' from 'gsplat' ，可以参考：https://github.com/nerfstudio-project/nerfstudio/issues/2727 ，大概就是要么两次卸载重装gsplat，要么重新打开终端。我没有卸载重装，重新打开终端就好了

如果 crypt.h不存在就`cp /usr/include/crypt.h /envs/nerfstudio/include/python3.8/`
### Installing nerfstudio

Easy option:

```bash
pip install nerfstudio
```

**OR** if you want the latest and greatest:

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

**OR** if you want to skip all installation steps and directly start using nerfstudio, use the docker image:

See [Installation](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md) - **Use docker image**.

## 2. Training your first model!

The following will train a _nerfacto_ model, our recommended model for real world scenes.

```bash
# Download some test data:
ns-download-data nerfstudio --capture-name=poster
# Train model
ns-train nerfacto --data data/nerfstudio/poster
```

If everything works, you should see training progress like the following:

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/3310961/202766069-cadfd34f-8833-4156-88b7-ad406d688fc0.png">
</p>

Navigating to the link at the end of the terminal will load the webviewer. If you are running on a remote machine, you will need to port forward the websocket port (defaults to 7007).

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/3310961/202766653-586a0daa-466b-4140-a136-6b02f2ce2c54.png">
</p>

### Resume from checkpoint / visualize existing run

It is possible to load a pretrained model by running

```bash
ns-train nerfacto --data data/nerfstudio/poster --load-dir {outputs/.../nerfstudio_models}
```




## 3. Exporting Results

Once you have a NeRF model you can either render out a video or export a point cloud.

### Render Video

First we must create a path for the camera to follow. This can be done in the viewer under the "RENDER" tab. Orient your 3D view to the location where you wish the video to start, then press "ADD CAMERA". This will set the first camera key frame. Continue to new viewpoints adding additional cameras to create the camera path. We provide other parameters to further refine your camera path. Once satisfied, press "RENDER" which will display a modal that contains the command needed to render the video. Kill the training job (or create a new terminal if you have lots of compute) and run the command to generate the video.

Other video export options are available, learn more by running

```bash
ns-render --help
```

### Generate Point Cloud

While NeRF models are not designed to generate point clouds, it is still possible. Navigate to the "EXPORT" tab in the 3D viewer and select "POINT CLOUD". If the crop option is selected, everything in the yellow square will be exported into a point cloud. Modify the settings as desired then run the command at the bottom of the panel in your command line.

Alternatively you can use the CLI without the viewer. Learn about the export options by running

```bash
ns-export pointcloud --help
```

## 4. Using Custom Data

Using an existing dataset is great, but likely you want to use your own data! We support various methods for using your own data. Before it can be used in nerfstudio, the camera location and orientations must be determined and then converted into our format using `ns-process-data`. We rely on external tools for this, instructions and information can be found in the documentation.

| Data                                                                                          | Capture Device | Requirements                                                      | `ns-process-data` Speed |
| --------------------------------------------------------------------------------------------- | -------------- | ----------------------------------------------------------------- | ----------------------- |
| 📷 [Images](https://docs.nerf.studio/quickstart/custom_dataset.html#images-or-video)          | Any            | [COLMAP](https://colmap.github.io/install.html)                   | 🐢                      |
| 📹 [Video](https://docs.nerf.studio/quickstart/custom_dataset.html#images-or-video)           | Any            | [COLMAP](https://colmap.github.io/install.html)                   | 🐢                      |
| 🌎 [360 Data](https://docs.nerf.studio/quickstart/custom_dataset.html#data-equirectangular)   | Any            | [COLMAP](https://colmap.github.io/install.html)                   | 🐢                      |
| 📱 [Polycam](https://docs.nerf.studio/quickstart/custom_dataset.html#polycam-capture)         | IOS with LiDAR | [Polycam App](https://poly.cam/)                                  | 🐇                      |
| 📱 [KIRI Engine](https://docs.nerf.studio/quickstart/custom_dataset.html#kiri-engine-capture) | IOS or Android | [KIRI Engine App](https://www.kiriengine.com/)                    | 🐇                      |
| 📱 [Record3D](https://docs.nerf.studio/quickstart/custom_dataset.html#record3d-capture)       | IOS with LiDAR | [Record3D app](https://record3d.app/)                             | 🐇                      |
| 📱 [Spectacular AI](https://docs.nerf.studio/quickstart/custom_dataset.html#spectacularai)    | IOS, OAK, [others](https://www.spectacularai.com/mapping#supported-devices) | [App](https://apps.apple.com/us/app/spectacular-rec/id6473188128) / [`sai-cli`](https://www.spectacularai.com/mapping) | 🐇 |
| 🖥 [Metashape](https://docs.nerf.studio/quickstart/custom_dataset.html#metashape)             | Any            | [Metashape](https://www.agisoft.com/)                             | 🐇                      |
| 🖥 [RealityCapture](https://docs.nerf.studio/quickstart/custom_dataset.html#realitycapture)   | Any            | [RealityCapture](https://www.capturingreality.com/realitycapture) | 🐇                      |
| 🖥 [ODM](https://docs.nerf.studio/quickstart/custom_dataset.html#odm)                         | Any            | [ODM](https://github.com/OpenDroneMap/ODM)                        | 🐇                      |
| 👓 [Aria](https://docs.nerf.studio/quickstart/custom_dataset.html#aria)                       | Aria glasses   | [Project Aria](https://projectaria.com/)                          | 🐇                      |
| 🛠 [Custom](https://docs.nerf.studio/quickstart/data_conventions.html)                        | Any            | Camera Poses                                                      | 🐇                      |


## 5. Advanced Options

### Training models other than nerfacto

We provide other models than nerfacto, for example if you want to train the original nerf model, use the following command

```bash
ns-train vanilla-nerf --data DATA_PATH
```

For a full list of included models run `ns-train --help`.

### Modify Configuration

Each model contains many parameters that can be changed, too many to list here. Use the `--help` command to see the full list of configuration options.

```bash
ns-train nerfacto --help
```

### Tensorboard / WandB / Viewer

We support four different methods to track training progress, using the viewer[tensorboard](https://www.tensorflow.org/tensorboard), [Weights and Biases](https://wandb.ai/site), and ,[Comet](https://comet.com/?utm_source=nerf&utm_medium=referral&utm_content=github). You can specify which visualizer to use by appending `--vis {viewer, tensorboard, wandb, comet viewer+wandb, viewer+tensorboard, viewer+comet}` to the training command. Simultaneously utilizing the viewer alongside wandb or tensorboard may cause stuttering issues during evaluation steps. The viewer only works for methods that are fast (ie. nerfacto, instant-ngp), for slower methods like NeRF, use the other loggers.

# Learn More

And that's it for getting started with the basics of nerfstudio.

If you're interested in learning more on how to create your own pipelines, develop with the viewer, run benchmarks, and more, please check out some of the quicklinks below or visit our [documentation](https://docs.nerf.studio/) directly.

| Section                                                                                  | Description                                                                                        |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [Documentation](https://docs.nerf.studio/)                                               | Full API documentation and tutorials                                                               |
| [Viewer](https://viewer.nerf.studio/)                                                    | Home page for our web viewer                                                                       |
| 🎒 **Educational**                                                                       |
| [Model Descriptions](https://docs.nerf.studio/nerfology/methods/index.html)              | Description of all the models supported by nerfstudio and explanations of component parts.         |
| [Component Descriptions](https://docs.nerf.studio/nerfology/model_components/index.html) | Interactive notebooks that explain notable/commonly used modules in various models.                |
| 🏃 **Tutorials**                                                                         |
| [Getting Started](https://docs.nerf.studio/quickstart/installation.html)                 | A more in-depth guide on how to get started with nerfstudio from installation to contributing.     |
| [Using the Viewer](https://docs.nerf.studio/quickstart/viewer_quickstart.html)           | A quick demo video on how to navigate the viewer.                                                  |
| [Using Record3D](https://www.youtube.com/watch?v=XwKq7qDQCQk)                            | Demo video on how to run nerfstudio without using COLMAP.                                          |
| 💻 **For Developers**                                                                    |
| [Creating pipelines](https://docs.nerf.studio/developer_guides/pipelines/index.html)     | Learn how to easily build new neural rendering pipelines by using and/or implementing new modules. |
| [Creating datasets](https://docs.nerf.studio/quickstart/custom_dataset.html)             | Have a new dataset? Learn how to run it with nerfstudio.                                           |
| [Contributing](https://docs.nerf.studio/reference/contributing.html)                     | Walk-through for how you can start contributing now.                                               |
| 💖 **Community**                                                                         |
| [Discord](https://discord.gg/uMbNqcraFc)                                                 | Join our community to discuss more. We would love to hear from you!                                |
| [Twitter](https://twitter.com/nerfstudioteam)                                            | Follow us on Twitter @nerfstudioteam to see cool updates and announcements                         |
| [Feedback Form](TODO)                                                                    | We welcome any feedback! This is our chance to learn what you all are using Nerfstudio for.        |

# Supported Features

We provide the following support structures to make life easier for getting started with NeRFs.

**If you are looking for a feature that is not currently supported, please do not hesitate to contact the Nerfstudio Team on [Discord](https://discord.gg/uMbNqcraFc)!**

- :mag_right: Web-based visualizer that allows you to:
  - Visualize training in real-time + interact with the scene
  - Create and render out scenes with custom camera trajectories
  - View different output types
  - And more!
- :pencil2: Support for multiple logging interfaces (Tensorboard, Wandb), code profiling, and other built-in debugging tools
- :chart_with_upwards_trend: Easy-to-use benchmarking scripts on the Blender dataset
- :iphone: Full pipeline support (w/ Colmap, Polycam, or Record3D) for going from a video on your phone to a full 3D render.

  # notes
  * 当你激活一个conda环境时，它会自动将该环境中的bin目录添加到系统的PATH变量中。因此，在这个环境中，你可以直接运行那些在bin目录中的可执行文件（例如ns-train）
  * 渲染器中各个相机视角的关闭在a6000/nerfstudio_mine/nerfstudio/viewer/viewer.py中的camera_handle = self.viser_server.scene.add_camera_frustum
  * 控制视角收缩远近的限制在a6000/nerfstudio_mine/nerfstudio/viewer/viewer.py中的VISER_NERFSTUDIO_SCALE_RATIO，越大，代表可以缩放的范围越近，参考10与100。且此参数不会影响任何训练，只影响查看
  * 提高渲染分辨率：https://github.com/nerfstudio-project/nerfstudio/issues/972
  * 更改viewer的查看范围：在crop viewpoint中进行更改，先enable，然后调整Crop scale，越大可以查看的范围越大，甚至能调到100\*100*100，如果勾选了split screen则是可以查看到一个彩色的正方形，在正方形内的才算是真正在去渲染的范围。
  * colmap时选择的图片多一点也许真能提高质量，选择了exhaustive应该不至于匹配不到点。
  * splatfacto训练快，viewer查看时渲染也快。splatfacto的模糊区域是尖庄模糊，nerfacto是糊状模糊。
  * controlpanel具体内容在viewer/control_panel.py中修改。要修改某个栏中具体有哪些控件，把add element注释掉就行。只通过控件的visible这时候可能不太行。
  * 控制拉伸范围时用VISER_NERFSTUDIO_SCALE_RATIO，但是把拉伸范围限制后，初始视角也会被限制。通过观察发现，在viewer.py中的handle_new_client函数初始时要执行十次渲染才会得到最终的那个背景图片（viser是在render_state_machine.py中通过设置背景图片来查看二维结果的）。于是可以设置一个is_first，设置为11。但是好像11倒计时结束后，它还是是会回去，因为相机位置依旧没变，当VISER_NERFSTUDIO_SCALE_RATIO变了场景自然就变了。
  * 还是没有解决怎么初始视角的问题。他这个写的很奇怪，大概就是在handle_new_client中激活渲染，然后get_camera_state决定视角，这个视角是从client中读到的，我也不知道在哪里初始化了client，应该client是属于viser server里面的，然后又是从client中的camera也就是camerahandle中读取当前屏幕的视角。我更改了默认的初始化值，但是依旧没有用。
  * viser文档写的很扯，看不明白，没办法

