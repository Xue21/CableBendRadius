# Calculating Cable Bend Radius Using Semantic Segmentation and a Depth Camera
It involves calculating cable bend radius based on vision using the **Intel RealSense D435i** depth camera, and building a user interface that integrates the camera and **MySQL database**.
## User Interface
![7](https://github.com/Xue21/CableBendRadius/assets/103324432/b1e42153-bf1b-4cf7-ad16-70f2b330ef21)

It comprises three functional modules: real-time data display, data storage and retrieval, and algorithmic prediction. 
 - **Real-time Data Display**: This module allows real-time display of RGB images captured by the depth camera, providing immediate visual feedback.
 - **Data Storage and Retrieval**: In this module, users can save and load desired images and depth information either locally or from a database.
 - **Cable Label OCR Detection**: It enables the system to detect cable type using Optical Character Recognition (OCR) algorithms and retrieve relevant information related to the identified cable type.
 - **Cable Bend Radius Detection**: This component performs calculations to determine the bend radius of cables and checks if they meet specified standards.
## Operating Instructions
These instructions should help you set up and run the following three Python scripts according to your specific camera and database configurations.
### 1.**wire_prediction.py**
After running, provide the RGB image file and its corresponding depth TXT file with the same name from your D435i depth camera as inputs. This script will process the data and perform cable bend radius prediction.
 ### **2.wire_control_ui.py**
Before running, update the database username and password in the code with your own credentials. This script doesn't require a connection to the D435i depth camera and will generate a user interface.
 ### **3.wire_control_ui_main.py**
Before running, connect to your D435i depth camera, modify the database username and password within the code to your own credentials. This script will create a user interface.
## Technical Route
### **General technical route**

![8](https://github.com/Xue21/CableBendRadius/assets/103324432/d8c68518-edbc-41f9-ad58-b8000c72137d)
### **Improved DeepLabv3+ semantic segmentation model**

![1](https://github.com/Xue21/CableBendRadius/assets/103324432/0ccf53bf-1a78-4a93-a8ff-62719c0c24ce)
### **Results**  
This is the result of the bending radius calculation of the cable I placed myself.  
  
<div style="display: flex;">  
    <img src="https://github.com/Xue21/CableBendRadius/assets/103324432/5b623de6-95c1-46bb-b435-a698c51bf17d" width="30%" />  
    <img src="https://github.com/Xue21/CableBendRadius/assets/103324432/1855e604-a3ea-4c89-ad0b-8637caa7f207" width="30%" />  
    <img src="https://github.com/Xue21/CableBendRadius/assets/103324432/8db33ac1-9549-4637-afa9-7231cf247c10" width="30%" />  
</div>
The middle image represents the semantic segmentation result, and the image on the right displays the calculation results, with each cable branch's result shown in the middle of its respective branch.
