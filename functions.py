from numpy import savez_compressed, asarray, load, expand_dims
from mtcnn.mtcnn import MTCNN
from os import listdir
from os.path import isdir
from PIL import Image
import cv2
from tqdm import tqdm
from augmentation import rewrite_to_augmented

def get_embedding(model, face_pixels):
    """Get the face embedding for one face"""
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

def face_to_embedings(faces, model):
    """Convert each face in the train set to an embedding."""
    embedings = []
    for face_pixels in faces:
        embedding = get_embedding(model, face_pixels)
        embedings.append(embedding)
    embedings = asarray(embedings)
    return embedings

def extract_face(filename):
    required_size=(160, 160)
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return asarray(image)

def load_dataset(directory):
    """Load a dataset that contains one subdir for each class that in turn contains images."""
    X, y = [], []
    # enumerate all folders named with class labels
    for subdir in tqdm(listdir(directory), desc="Loading dataset"):
        path = directory + subdir + '\\'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

def load_faces(directory):
    """
    Load images and extract faces for all images in a directory
    """
    faces = []
    # enumerate files
    for filename in listdir(directory):
        path = directory + filename
        # get face or augment it
        face = extract_face(path)
        if face is None:
            print(f'I can`t find a person in {filename}!\nI will try to use augmentation.\n')
            back = cv2.imread('backg.jpg')
            rewrite_to_augmented(path, back)
            continue
        faces.append(face)
    return faces