# Calculating Cable Bend Radius Using Semantic Segmentation and a Depth Camera
It involves calculating cable bend radius based on vision using the **Intel RealSense D435i** depth camera, and building a user interface that integrates the camera and **MySQL database**.
## User Interface
![7](https://github.com/Xue21/CableBendRadius/assets/103324432/585ef043-198c-41b8-b8be-035eec9a892d)

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

![8](https://github.com/Xue21/CableBendRadius/assets/103324432/a508ca2d-603a-44e5-8429-2612f1286a35)
### **Improved DeepLabv3+ semantic segmentation model**

![1](https://github.com/Xue21/CableBendRadius/assets/103324432/2ebe3f87-1a17-402d-8e9c-ec62dd84eae8)
### **Results**  
This is the result of the bending radius calculation of the cable I placed myself.  
  
<div style="display: flex;">  
    <img src="https://github.com/Xue21/CableBendRadius/assets/103324432/e4dfcfdd-cb9d-4e96-a305-ac3c29e39c21" width="30%" />  
    <img src="https://github.com/Xue21/CableBendRadius/assets/103324432/288b9829-163b-49f0-8e28-b428f4bbca0c" width="30%" />  
    <img src="https://github.com/Xue21/CableBendRadius/assets/103324432/ee4d4a3f-b12c-4749-9910-42ff49af96c2" width="30%" />  
</div>
The middle image represents the semantic segmentation result, and the image on the right displays the calculation results, with each cable branch's result shown in the middle of its respective branch.


