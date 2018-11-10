# This folder is for my personal and educational projects that do not deserve their separate folder.
## fcnn.py 
My take on using Fully Connected Neural Network for classifying the famous IRIS flower database. Two hidden layers with ReLU function,
the last layer scales the output using Sigmoid and gives classification. MSE function was chosen as a cost function. PyTorch framework was used 
for this purely _educational_ project. 

How to use: make sure that you have iris.data file from http://archive.ics.uci.edu/ml/machine-learning-databases/iris/ at the same folder as the .py file and just launch the code. It will take time to train. 

This particular version takes _all_ the database to train. The version where it splits the database and checks the testing set will not be uploaded any time soon for logistical reasons. Also I'm lazy. 
