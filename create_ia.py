import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import PIL

# Parâmetros
img_height, img_width = 224, 224
batch_size = 32
epochs = 10

# Definição das classes
classes = {0: 'Alicate', 1: 'cabo', 2: 'carrinho', 3: 'chave de fenda', 4: 'chave estrela', 5: 'serra'}
num_classes = len(classes)

# Configuração do gerador de imagens
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'Treino/Treine/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Múltiplas classes
)

validation_generator = validation_datagen.flow_from_directory(
    'Treino/Validation/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Múltiplas classes
)

# Construção do modelo com softmax e múltiplas classes
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # Ajuste para várias classes
])

# Compilação do modelo com categorical_crossentropy
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Salvando o modelo em formato .h5
model.save('ia.h5')
print("Modelo salvo como 'ferramenta_classificador.h5'")
