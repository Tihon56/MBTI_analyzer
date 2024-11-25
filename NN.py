import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv('prepared_data.csv')

#Разделяю датасет, на тренировочный и практический для оценки точности
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Personality", axis = 1),  
    data["Personality"], 
    test_size=0.2,                              
    random_state=42                            
)


X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
y_train = y_train.astype('float32') 
y_test = y_test.astype('float32')

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




#размерность 128061*9

model = tf.keras.Sequential([
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(64),
    layers.Dense(17, activation='softmax')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callback для остановки при отсутствии улучшений
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

# Обучение модели
model.fit(X_train, y_train, epochs=1000, validation_split=0.2, callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")


# Визуализация распределения типов личности
personality_counts = data["Personality"].value_counts()
labels = [f"Тип {i}" for i in range(1, 17)]
plt.bar(labels, personality_counts)
plt.xlabel("Типы личности")
plt.ylabel("Количество")
plt.title("Распределение типов личности")
plt.xticks(rotation=45)
plt.show()