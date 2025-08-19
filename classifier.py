import os
import warnings
warnings.filterwarnings("ignore")
import datetime
import time
from numpy import load
from keras.models import load_model, Sequential, save_model
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, Normalizer
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

current_file_dir = os.path.dirname(os.path.abspath(__file__))

class Trainer:

    def __init__(self, project_dirpath):
        self.faces_embeddings = os.path.join(project_dirpath, "file/ce_embedding.npz")
        self.dense_classifier = os.path.join(project_dirpath, "file/classifier.h5")
        
    def build_dense_model(self, input_shape, num_classes):
        """Build a simple Dense model for classification"""
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.4))
        model.add(Dense(num_classes, activation='softmax'))
        # compile model
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def classifier(self):
        """Create a Dense Classifier for the Faces Dataset"""
        # load dataset
        data = load(self.faces_embeddings)
        train_X, train_y, test_X, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print(f'Dataset: train={train_X.shape[0]}, test={test_X.shape[0]}')

        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        train_X = in_encoder.transform(train_X)
        test_X = in_encoder.transform(test_X)

        # label encode targets
        out_encoder = LabelEncoder()
        out_encoder.fit(train_y)
        train_y = out_encoder.transform(train_y)
        test_y = out_encoder.transform(test_y)

        # one-hot encode targets
        train_y = to_categorical(train_y)
        test_y = to_categorical(test_y)

        # build Dense model
        input_shape = (train_X.shape[1],)
        model = self.build_dense_model(input_shape, train_y.shape[1])
        print('Dense Model Built')
        num_classes = train_y.shape[1]
        # Define class weights to penalize the positive class more heavily
        #class_weights = {i: 1.0 for i in range(num_classes)} 
        # fit model
        history = model.fit(train_X, train_y, epochs=35, batch_size=16, validation_data=(test_X, test_y))

        # save the model to disk
        model.save(self.dense_classifier)

        # Plot training history
        self.plot_training_history(history)

        # predict
        yhat_train = model.predict(train_X)
        yhat_test = model.predict(test_X)

        # decode predictions
        yhat_train = yhat_train.argmax(axis=1)
        yhat_test = yhat_test.argmax(axis=1)

        # score
        score_train = accuracy_score(out_encoder.inverse_transform(yhat_train), out_encoder.inverse_transform(train_y.argmax(axis=1)))
        score_test = accuracy_score(out_encoder.inverse_transform(yhat_test), out_encoder.inverse_transform(test_y.argmax(axis=1)))

        # summarize
        print(f'Accuracy: train={score_train * 100:.3f}, test={score_test * 100:.3f}')
        print(classification_report(out_encoder.inverse_transform(test_y.argmax(axis=1)), out_encoder.inverse_transform(yhat_test)))
        
        return
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Classification Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Classification Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        plt.show()
    
    def predict_with_threshold(self, model, data, threshold=0.5):
        """Predict with a probability threshold"""
        predictions = model.predict(data)
        max_probabilities = predictions.max(axis=1)
        predicted_classes = predictions.argmax(axis=1)
        filtered_predictions = [pred if max_prob > threshold else -1 for pred, max_prob in zip(predicted_classes, max_probabilities)]
        return filtered_predictions

    def start(self):
        start_time = time.time()
        st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print(f"Face trainer Initiated at {st}")
        print("-----------------------------------------------------------------------------------------------")
        self.classifier()
        end_time = time.time()
        et = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        print("-----------------------------------------------------------------------------------------------")
        print(f"Face trainer Completed at {et}")
        print(f"Total time Elapsed {round(end_time - start_time)} secs")
        print("-----------------------------------------------------------------------------------------------")

        return
    
if __name__ == "__main__":
    facetrainer = Trainer(current_file_dir)
    facetrainer.start()
