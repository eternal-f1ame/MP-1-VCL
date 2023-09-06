import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch.nn as nn

class NewModel(nn.Module):

    def __init__(self): 
        super(NewModel, self).__init__()    

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)        
        self.relu2 = nn.ReLU()        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)       
        self.relu3 = nn.ReLU()        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)              # Fully connected layers        
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # 16x16 is the feature map size after 3 pooling layers        
        self.relu4 = nn.ReLU()    
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 5)  

    def forward(self, x):        
        x = self.pool1(self.relu1(self.conv1(x)))       
        x = self.pool2(self.relu2(self.conv2(x)))        
        x = self.pool3(self.relu3(self.conv3(x)))       
        x = x.view(x.size(0), -1)  # Flatten the tensor   
        x = self.dropout1(x)     
        x = self.relu4(self.fc1(x)) 
        x = self.fc2(x)        
        return x
        
transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model = NewModel()
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()
def predict(image):
    image = Image.open(image)
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

predictions_map = {
    0: 'Bracelet',
    1: "Eyeglasses",
    2: "Floppers",
    3: "Ring",
    4: "Watch"
    }

st.title("Wearable Accessories Classification")
st.write("A minimal classifier for wearable accessories using PyTorch and Streamlit can classify between 5 classes of wearable accessories - [ Bracelet, Eyeglasses, Floppers, Ring, Watch ].")
file = st.file_uploader("Upload Inference Image", type=["jpg", "png", "jpeg"])
if file:
    image = Image.open(file)
    st.image(image, width=300)
    with st.spinner("Fetching Results..."):
        predictions = predict(file)
        print(type(predictions))
        st.write(predictions_map[predictions])
