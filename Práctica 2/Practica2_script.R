## -----------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2017-2018
## Juan Gómez Romero
## Ejemplo basado en 'Deep Learning with R'
## -----------------------------------------------------------------------------------

install.packages("ggplot2")
library(ggplot2)
install.packages("keras")
library(keras)
install_keras(method = "conda",tensorflow = "gpu")
#install_keras(method = "conda",tensorflow = "cpu")

## ----------------------------------------------------------------------------------
## Cargar y pre-procesar imágenes
## ----------------------------------------------------------------------------------

train_dir      <- './Practica 2 SIGE/dataset2-master/images/TRAIN/' 
validation_dir <- './Practica 2 SIGE/dataset2-master/images/TEST_SIMPLE/'
test_dir       <- './Practica 2 SIGE/dataset2-master/images/TEST/'


img_sample <- image_load(path = './Practica 2 SIGE/dataset2-master/images/TRAIN/EOSINOPHIL/_0_207.jpeg', target_size = c(150, 150))
img_sample_array <- array_reshape(image_to_array(img_sample), c(1, 150, 150, 3))
plot(as.raster(img_sample_array[1,,,] / 255))


train_datagen      <- image_data_generator(rescale = 1/255) 
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen       <- image_data_generator(rescale = 1/255)


n_batch_size = 32

train_data <- flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen,
  target_size = c(150, 150),        # (w, h) -> (150, 150)
  batch_size = n_batch_size,        # grupos de 32 imágenes
  class_mode = "categorical"        # etiquetas categorical
)

validation_data <- flow_images_from_directory(
  directory = validation_dir,
  generator = validation_datagen,
  target_size = c(150, 150),        # (w, h) -> (150, 150)
  batch_size = n_batch_size,        # grupos de 32 imágenes
  class_mode = "categorical"        # etiquetas categorical
)

test_data <- flow_images_from_directory(
  directory = test_dir,
  generator = test_datagen,
  target_size = c(150, 150),        # (w, h) -> (150, 150)
  batch_size = n_batch_size,        # grupos de 32 imágenes
  class_mode = "categorical"        # etiquetas categorical
)

## ----------------------------------------------------------------------------------
## Crear modelo
## ----------------------------------------------------------------------------------

# Definición de la arquitectura
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = c(150,150, 3)) %>%
  layer_activation('relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu") %>% 
  layer_activation('relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% 
  layer_activation('relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 4, activation = "softmax")

summary(model)

# Compilación del modelo
model %>% compile(
  loss = "mean_squared_logarithmic_error",
  optimizer = 'adadelta',
  #optimizer = 'adam',
  #optimizer = 'adagrad',
  #optimizer = 'sgd',
  #optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy")
)

summary(model)

# Entrenamiento
n_epochs = 30

history <- model %>% 
  fit_generator(
    train_data,
    steps_per_epoch = 250,
    epochs = n_epochs,
    validation_data = validation_data,
    validation_steps = 75
  )

# Visualizar entrenamiento
plot(history)
history

# Guardar modelo (HDF5)
model %>% save_model_hdf5("practica2_30epocs_meanlog_adadelta_256.h5")

# Evaluar modelo
model %>% evaluate_generator(test_data, steps = 75)

## ----------------------------------------------------------------------------------
## Data augmentation
## ----------------------------------------------------------------------------------

data_augmentation_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

train_augmented_data <- flow_images_from_directory(
  directory = train_dir,
  generator = data_augmentation_datagen,  # ¡usando nuevo datagen!
  target_size = c(150, 150),              # (w, h) -> (150, 150)
  batch_size = 32,                        # grupos de 32 imágenes
  class_mode = "categorical"              # etiquetas categorical
)

history <- model %>% 
  fit_generator(
    train_augmented_data,
    steps_per_epoch = 250,
    epochs = n_epochs,
    validation_data = validation_data,
    validation_steps = 75
  )

# Visualización del entrenamiento
plot(history)
history

# Guarda modelo (HDF5)
model %>% save_model_hdf5("test_practica2_augmentation_30epcs_adadelta_256.h5")

# Evaluación del modelo
model %>% evaluate_generator(test_data, steps = 75)
