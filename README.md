# FSOD Dashboard
FSOD stands for Firearms and Sharp Object Detector.  Conclusively, this dashboard is a web application made with streamlit that can detect several kind of firearms and sharp object threat that I build for my bachelor's thesis project.  Object detection algorithm used to make the  model are YOLO-R and also used Deepsort for tracking purpose.

## Detectable Classes
Classes that are available for detection in this web application are as follows:
1. Pistol
2. Senapan Serbu (Assault Rifle)
3. Pisau (Knife)
4. Celurit (Sickle)
5. Kapak (Axe)
6. Bukan Senjata (Not Weapon) [Consist of some handheld items such as smartphone, wallet, cash money, and ATM card]

## Try Out Application
Follow this steps below in order to try this locally:
1. git clone https://github.com/blitzkz23/fsod-dashboard
2. pip install requirements.txt
3. pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html (dependencies for pytorch with CUDA)
4. streamlit run fsod_dashboard.py

## Notebook I Used to Train YOLOR Model
Check it out on the link below:

https://colab.research.google.com/drive/15VnLzKiTDnXx-SOZNCFWy7V5ffFaGHnx#scrollTo=c0NuvEqB-IK2
