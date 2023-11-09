# Calculating Cable Bend Radius Using Semantic Segmentation and a Depth Camera
It involves calculating cable bend radius based on vision using the **Intel RealSense D435i** depth camera, and building a user interface that integrates the camera and **MySQL database**.
## User Interface
It comprises three functional modules: real-time data display, data storage and retrieval, and algorithmic prediction. 
 - **Real-time Data Display**: This module allows real-time display of RGB images captured by the depth camera, providing immediate visual feedback
 - **Data Storage and Retrieval**: In this module, users can save and load desired images and depth information either locally or from a database.
 - **Cable Label OCR Detection**: It enables the system to detect cable type using Optical Character Recognition (OCR) algorithms and retrieve relevant information related to the identified cable type.
 - **Cable Bend Radius Detection**: This component performs calculations to determine the bend radius of cables and checks if they meet specified standards
## Operating Instructions
These instructions should help you set up and run the following three Python scripts according to your specific camera and database configurations.
### 1.**wire_prediction.py**
After running, provide the RGB image file and its corresponding depth TXT file with the same name from your D435i depth camera as inputs. This script will process the data and perform cable bend radius prediction.
 ### **2.wire_control_ui.py**
Before running, update the database username and password in the code with your own credentials. This script doesn't require a connection to the D435i depth camera and will generate a user interface.
 ### **3.wire_control_ui_main.py**
Before running, connect to your D435i depth camera, modify the database username and password within the code to your own credentials. This script will create a user interface.
## Technical Route
### General technical route  
  
![8](https://github.com/Xue21/CableBendRadius/assets/103324432/f53c53fc-a1e9-4ded-9a37-831771f86b4d)  
### Improved DeepLabv3+ semantic segmentation model

![1](https://github.com/Xue21/CableBendRadius/assets/103324432/e04a8a46-61dd-4549-9524-d1e0dc924ccf)  
### Results  
This is the result of the bending radius calculation of the cable I placed myself.  
  
<div style="display: flex;">  
    <img src="https://github.com/Xue21/CableBendRadius/assets/103324432/ebb0a942-4957-4a98-bb19-b8a7d14cdd6a" width="30%" />  
    <img src="https://github.com/Xue21/CableBendRadius/assets/103324432/d462c82a-3848-4290-9cff-7805fd47772d" width="30%" />  
    <img src="https://github.com/Xue21/CableBendRadius/assets/103324432/d7d2492b-f78f-44b1-a7b6-6c4d28f70d17" width="30%" />  
</div>  
  
The middle image represents the semantic segmentation result, and the image on the right displays the calculation results, with each cable branch's result shown in the middle of its respective branch.