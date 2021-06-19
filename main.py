from nNet import cost_function, nNetwork
import random 

'''
Inputs: Date Month and Temprature data of lasst five days
Output: Tempreture of Tommrrow
Layer: [7,7,3,2,1]
'''
# Defining Model
model = nNetwork([7, 7, 3, 2, 1])

# Loading Data
x_train = []
y_train = []
x_test = []
y_test = []

with open("data.txt", "r") as data_file:
    lines = data_file.read().split("\n")
    training_len = int(len(lines)*.8)
    for i in range(5,len(lines)-1):
        line = lines[i].split("         ")
        month = int(line [0])
        day = int(line[1])
        ip = [day,month]
        for j in range(i-5,i):
            ip.append(float(lines[j].split("         ")[-1]))
        op=float(lines[i].split("         ")[-1])
        if i<training_len:
            x_train.append(ip)
            y_train.append([op/100])
        else:
            x_test.append([ip])
            y_test.append([op/100])  

#Taining For 3 Epochs                
for epoch in range(3):
    cost=0 
    for i in range(len(y_train)):
        model.feedforward(x_train[i])
        cost+=cost_function(y_train,model.predict(x_train[i]))
        model.back_prop(y_train[i])
    print(f"Epoch {epoch+1}\t Cost: {round(float(cost)/len(x_train),2)}")
    model.save(float(cost)/len(x_train))

#Testing 
start=random.randint(0,len(x_test)-11)
for i in range(start,start+10):
    preds = model.predict(x_test[i][0])
    print(f"Expected: {y_test[i][0]*100} Predicted: {round(float(preds)*100,2)}")
input("Press Enter To Exit")