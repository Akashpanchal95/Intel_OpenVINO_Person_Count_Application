# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

In this project I have used my 2 year old model, which is trained on the faster_rcnn_inception_v2 model, for dataset perspective I have collected the person images from internet, mobile and other sources. Here I have converted the model using OpenVINO™ which is provide the model converter steps under the model optimizer.

Below you can find drive link for my custom model with the custom dataset, the model accuracy is two low so we need to set  prob threshold ".1" for better result.
https://drive.google.com/drive/folders/1BA_8PTuQfG-9riRHwGtw5iKoNFilPpaC?usp=sharing 


Download the model from google drive and pass PB model and config file for model conversion.

To convert the original model to IR using the following command (i.e. PB to IR):

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --tensorflow_object_detection_api pipeline.config
```

When above command its complete it will give folllowing message.
```
XML File:
[ SUCCESS ] XML file: /home/user_path//./frozen_inference_graph.xml
[ SUCCESS ] BIN file: /home/user_path/./frozen_inference_graph.bin
```

To Run the project using following command 

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model_path.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.1 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were IR model give best performance compare to original model.

You can find the model details in below table


Here I have Comparing the two models i.e. ssd_inception_v2_coco and faster_rcnn_inception_v2_custom model in terms of latency and memory. here we can clearly see that Latency(microseconds) and Model memory both are decreases using OpenVINO as compare to original TensorFlow model.

| Model/Framework                             		  | Latency (microseconds)            | Memory (Mb) |
| -----------------------------------                 |:---------------------------------:| -------:	|
| faster_rcnn_inception_v2_custom_dataset (Tensorflow)| 690                               | 57.2   |
| faster_rcnn_inception_v2_custom_dataset (OpenVINO)  | 230                               | 53.2   |
| ssd_inception_v2_coco (tensorflow)          		  | 580                               | 538    |
| ssd_inception_v2_coco (OpenVINO)            		  | 150                               | 329    |


## Assess Model Use Cases

This application could keep checking on a number of people in a particular area(i.e. any restricted area) Example as ATM, industrial area, as a current crisis(i.e. COVID-19) to count the number of people in the frame.
Each of these use cases would be useful because we can easily deployed into the edge devices.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows,
If lighting condition change frequently then model gives a false positive or even sometimes not detected due to fewer data. 
Camera angle its also play important, because due to the occlusion problem the two-person counted as one, so it will give false result in counting. and also camera have enough resolution.(i.e. compitable with the camera trained model)

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
