import os
import warnings

warnings.filterwarnings("ignore")
from numpy import savez_compressed, asarray, load, expand_dims
from keras.models import load_model,Model

current_file_dir = os.path.dirname(os.path.abspath(__file__))

class FaceTrainer:

    def __init__(self, project_dirpath):
        self.faces_npz = os.path.join(project_dirpath, "file/ce_dataset.npz")
        self.model_path = os.path.join(project_dirpath, "file/My_Model.h5")
        self.faces_embeddings = os.path.join(project_dirpath, "file/ce_embedding.npz")
        # Load the Keras model
        self.model = load_model(self.model_path)
        # Create a new model that outputs the features from the 'Bottleneck' layer
        self.feature_extractor = Model(inputs=self.model.input, outputs=self.model.get_layer('dropout').output)
        
    def create_faces_embedding_npz(self):
        """Create npz file for all the face embeddings in train_dir, val_dir"""
        data = load(self.faces_npz)
        train_X, train_y, test_X, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('Loaded: ', train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        # convert each face in the train set to an embedding
        newTrainX = self.face_to_embedings(train_X)
        newTestX = self.face_to_embedings(test_X)
        # save arrays to one file in compressed format
        savez_compressed(self.faces_embeddings, newTrainX, train_y, newTestX, test_y)
        return

    def face_to_embedings(self, faces):
        """Convert each face in the train set to an embedding."""
        embedings = []
        for face_pixels in faces:
            embedding = self.get_embedding(face_pixels)
            embedings.append(embedding)
        embedings = asarray(embedings)
        return embedings

    def get_embedding(self, face_pixels):
        """Get the face embedding for one face"""
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = self.feature_extractor.predict(samples)
        return yhat[0]

    def start(self):
        # Get embeddings for all the extracted faces
        self.create_faces_embedding_npz()
        return

if __name__ == "__main__":
    facetrainer = FaceTrainer(current_file_dir)
    facetrainer.start()
