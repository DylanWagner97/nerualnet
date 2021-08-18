from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from numpy import loadtxt
#import data and set to x for input and y for output

#import out dataset in this case breasttissue
dataset = loadtxt('breasttissue_train.csv', delimiter=",")
x = dataset[:,1:14]
y = dataset[:,0]
#encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def get_model():
    #create model for our network
    model = Sequential()
    #Adds the first hidden layer
    model.add(Dense(13, input_dim=9, activation='relu'))
    #second hidden layer and output
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='softmax'))
   
    # Compile model using adam for best first and second prediction
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #scale data for better perfromance 
    scaler = MinMaxScaler()
    scaleVal = scaler.fit_transform(x)
    #fit the model to our data 
    model.fit(scaleVal, dummy_y, epochs=150, batch_size=10)
    return model

get_model()
