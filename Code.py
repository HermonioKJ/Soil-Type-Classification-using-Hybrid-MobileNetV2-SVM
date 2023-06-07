#Extracting zip file
import zipfile
import os

file = 'new.zip' //dataset in zip form

with zipfile.ZipFile(file, 'r') as zip_ref:
    zip_ref.extractall('./')
os.remove(file)

#Model Training
import os
import numpy as np
from sklearn import svm
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataset_dir = 'new'


image_size = (224, 224) 


cnn_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


def extract_features(image_path):
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0 
    features = cnn_model.predict(image)
    features = np.reshape(features, (features.shape[0], -1))
    return features


features = []
labels = []
#Data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=20, 
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,  
    fill_mode='nearest'  
)

for class_folder in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_folder)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image_features = extract_features(image_path)
            features.append(image_features)
            labels.append(class_folder)

            image = load_img(image_path, target_size=image_size)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            augmented_images = data_augmentation.flow(image, batch_size=8)
            for i, augmented_image in enumerate(augmented_images):
                augmented_image_features = cnn_model.predict(augmented_image / 255.0)
                augmented_image_features = np.reshape(augmented_image_features, (augmented_image_features.shape[0], -1))
                features.append(augmented_image_features)
                labels.append(class_folder)
                if i == 10: #Number of copies each image will be augmented
                    break

X = np.concatenate(features)
y = np.array(labels)
classifier = svm.SVC(kernel='linear', C=0.01, probability = True)
classifier.fit(X, y)

class_descriptions = {
    "Loam":"Loam comprises three different materials: silt, clay, and sand. \nThe variety in particle sizes creates openings in the ground that allow air, water, and roots to pass through freely. \nLoam doesn’t dry too fast; it is soft and almost effortless to till.",
    "clay":"Clay’s heavy and compact structure holds moisture well and is ideal for moisture-loving plants. \nMany crops will thrive in this type of soil due to the high nutrient content. \nMeanwhile, clay is frequently alkaline which stops plants from getting all the nutrients they require to flourish and produce a high yield.",
    "sand":"The main advantage of the sandy type is that it is suitable for early planting because it is the first to warm up after winter. \nIt is not too prone to erosion and is effortless to till due to the large size of the particles.\nAdditionally, the sandy type is often acidic, meaning it has a low pH level. \nPlants growing in sandy soils may thus be deficient in the nutrients and moisture necessary for their growth.",
    "silt":"Silty ground particles have physical properties somewhere between those of sand and clay.\nBecause of its fine texture, silt holds more water than sand.\nSilty types of soil are fertile and contain a sufficient number of nutrients. \nMost plants will thrive when the drainage system is channelized correctly for silt.",
}

plant_loam = ["Root crops", "Wheat", "Cotton", "Sugar cane", "Most fruits", "Berries", "Climbing plants", "Flowers including roses, irises, gladiolus, and lilies"]
plant_clay = ["Vegetables: Broccoli, Cauliflower, Kale, Peas, Potatoes, Cabbage, And Brussels Sprouts","leafy Crops","fruit Trees","perennials", "Ornamental Plants", "Shrubs: Including Aster, Helen’s Flower, And Flowering Quince."]
plant_sand = ["Commercially Cultivated Plants: Collard Greens, Tomatoes, Melons, Squash, Strawberries, Sugarbeet, Lettuce, And Peppers", "Maize", "Millet", "Barley", "Root Vegetables: Potatoes, Parsnips, And Carrots", "Shrubs And Bulbs: Tulips, Tree Mallow, Sun Roses, And Hibiscus", "Herbs Native To Mediterranean Climates: Oregano, Rosemary, And Lavender."]
plant_silt = ["Most Vegetables", "Climbing Plants", "Perennials", "Grasses", "Shrubs", "Trees: Including Willow, Birch, And Dogwood."]

#Validating model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.model_selection import cross_val_score

#K-cross validation
k = 5 
scores = cross_val_score(classifier, X, y, cv=k)

for fold, score in enumerate(scores):
    print(f"Fold {fold + 1}: Accuracy = {score}")

mean_accuracy = np.mean(scores)
print(f"Mean Accuracy: {mean_accuracy}")

#Confusion Matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = classifier.predict(X)

cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#Test with your own image
test_image_path = 'loamm.jpg' //Image you want to test it with 
test_features = extract_features(test_image_path)
prediction_probs = classifier.predict_proba(test_features)
prediction_percentages = [f"{prob * 100:.2f}%" for prob in prediction_probs[0]]
predicted_label = classifier.predict(test_features)[0]

for label, prob in zip(classifier.classes_, prediction_probs[0]):
    prediction_percentage = prob * 100
    print(f"Soil Type: {label}")
    print(f"Prediction Percentage: {prediction_percentage:.2f}%")
predicted_description = class_descriptions[predicted_label]
print()
print(f"The soil type is predicted to be {predicted_label}")
print("Description:", predicted_description)
print()
if predicted_label == "Loam":
  print("Plants that are suitable to this soil are:")
  for i, plant in enumerate(plant_loam,1):
    print(f"{i}.{plant}")
if predicted_label == "clay":
  print("Plants that are suitable to this soil are:")
  for i, plant in enumerate(plant_clay,1):
    print(f"{i}.{plant}")
if predicted_label == "sand":
  print("Plants that are suitable to this soil are:")
  for i, plant in enumerate(plant_sand,1):
    print(f"{i}.{plant}")
if predicted_label == "silt":
  print("Plants that are suitable to this soil are:")
  for i, plant in enumerate(plant_silt,1):
    print(f"{i}.{plant}")

#Run it using anvil
pip install anvil-uplink

import anvil.server

anvil.server.connect("server_OB6GRHZKSRRZXC7K2F2MS4SE-BGFYVA65AO2TGXZR")

import requests
from io import BytesIO
import anvil.server
import anvil.media
@anvil.server.callable

def model_run(path):
    with anvil.media.TempFile(path) as filename:
      test_features = extract_features(filename)
      predicted_label = classifier.predict(test_features)[0]
      return predicted_label

@anvil.server.callable
def model_percent(path):
    with anvil.media.TempFile(path) as filename:
      test_features = extract_features(filename)
      prediction_probs = classifier.predict_proba(test_features)
      prediction_percentages = [f"{prob * 100:.2f}%" for prob in prediction_probs[0]]
      result = f"Loam: {prediction_percentages[0]} // Clay: {prediction_percentages[1]} // Sand: {prediction_percentages[2]} // Silt: {prediction_percentages[3]}"
      return result

@anvil.server.callable
def description(label):
  
  class_descriptions = {
    "Loam":"Loam comprises three different materials: silt, clay, and sand. \nThe variety in particle sizes creates openings in the ground that allow air, water, and roots to pass through freely. \nLoam doesn’t dry too fast; it is soft and almost effortless to till.",
    "clay":"Clay’s heavy and compact structure holds moisture well and is ideal for moisture-loving plants. \nMany crops will thrive in this type of soil due to the high nutrient content. \nMeanwhile, clay is frequently alkaline which stops plants from getting all the nutrients they require to flourish and produce a high yield.",
    "sand":"The main advantage of the sandy type is that it is suitable for early planting because it is the first to warm up after winter. \nIt is not too prone to erosion and is effortless to till due to the large size of the particles.\nAdditionally, the sandy type is often acidic, meaning it has a low pH level. \nPlants growing in sandy soils may thus be deficient in the nutrients and moisture necessary for their growth.",
    "silt":"Silty ground particles have physical properties somewhere between those of sand and clay.\nBecause of its fine texture, silt holds more water than sand.\nSilty types of soil are fertile and contain a sufficient number of nutrients. \nMost plants will thrive when the drainage system is channelized correctly for silt.",
  }
  predicted_description = class_descriptions[label]
  return predicted_description

@anvil.server.callable
def plant(predicted_label):
  
  plant_loam = ["Root crops", "Wheat", "Cotton", "Sugar cane", "Most fruits", "Berries", "Climbing plants", "Flowers including roses, irises, gladiolus, and lilies"]
  plant_clay = ["Vegetables: Broccoli, Cauliflower, Kale, Peas, Potatoes, Cabbage, And Brussels Sprouts","leafy Crops","fruit Trees","perennials", "Ornamental Plants", "Shrubs: Including Aster, Helen’s Flower, And Flowering Quince."]
  plant_sand = ["Commercially Cultivated Plants: Collard Greens, Tomatoes, Melons, Squash, Strawberries, Sugarbeet, Lettuce, And Peppers", "Maize", "Millet", "Barley", "Root Vegetables: Potatoes, Parsnips, And Carrots", "Shrubs And Bulbs: Tulips, Tree Mallow, Sun Roses, And Hibiscus", "Herbs Native To Mediterranean Climates: Oregano, Rosemary, And Lavender."]
  plant_silt = ["Most Vegetables", "Climbing Plants", "Perennials", "Grasses", "Shrubs", "Trees: Including Willow, Birch, And Dogwood."]
  if predicted_label == "Loam":
    print("Plants that are suitable to this soil are:")
    sentence = '\n'.join(str(num) for num in plant_loam)
  if predicted_label == "clay":
    print("Plants that are suitable to this soil are:")
    sentence = '\n'.join(str(num) for num in plant_clay)
  if predicted_label == "sand":
    print("Plants that are suitable to this soil are:")
    sentence = '\n'.join(str(num) for num in plant_sand)
  if predicted_label == "silt":
    print("Plants that are suitable to this soil are:")
    sentence = '\n'.join(str(num) for num in plant_silt)
  return sentence

import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
@anvil.server.callable
def store_file_in_table(file_content):
    app_tables.inputs.add_row(content=file_content)

@anvil.server.callable
def search_table(search_query):
    return app_tables.inputs.search(search_query)

anvil.server.wait_forever()
