library(tidyverse)
library(funModeling)
library(ggplot2)
library(Hmisc)
library(corrplot)

## ---------------------------------------------------------------
## 1. Lectura de datos  ## Eliminando NA y NULL 
data_raw <- read_csv('LoanStats_2017Q4.csv', na = c('NA', 'n/a', '', ' ')) 
glimpse(data_raw) #Valores tipos de datos  
# Estado del Dataset, Tabla que especifica cada variable. Guardamos la salidad de df_status en variable "status"
dim(data_raw)
status <- df_status(data_raw)#El estado del dataset  

## ---------------------------------------------------------------
## 2. Eliminar columnas no Utiles

# Identificar columnas con mas del 90% de los valores a 0
zero_cols <- status %>%
  filter(p_zeros > 90) %>%
  select(variable)

hist(status$p_zeros, 
     main="Columnas con más del 90% de los valores a 0", 
     xlab="p_zeros", 
     border="blue", 
     col="green",
     las=1, 
     breaks=5
)

zero_cols

# Identificar columnas con mas del 50% de los valores a NA
na_cols <- status %>%
  filter(p_na > 50) %>%
  select(variable)

hist(status$p_na, 
     main="Columnas con más del 90% de los valores a 0", 
     xlab="p_na", 
     border="green", 
     col="blue",
     las=1, 
     breaks=5
)

table(na_cols)
na_cols

# Identificar columnas con <= 1 valores diferentes
eq_cols <- status %>%
  filter(unique <= 1) %>%
  select(variable)

eq_cols

# Identificar columnas >75% valores diferentes  #con 50 son suficientes
dif_cols <- status %>%
  filter(unique > 0.75 * nrow(data_raw)) %>%  ###Filtrando el numero de filas (Numero de posibles valores diferentes)
  select(variable)

dif_cols

# Junta varias de las columnas Inservibles 
## Me permite hacer una union de varios dataframes
remove_cols <- bind_rows(
    list(
      zero_cols,
      na_cols,
      eq_cols,
      dif_cols
    )
  )

# Elimina las columnas Inservibles
data <- data_raw %>%
    select(-one_of(remove_cols$variable))

# Grafica de la variable loan_status sin filtrar
ggplot(data) +
    geom_histogram(aes(x = loan_status, fill = loan_status), stat= 'count')

## ---------------------------------------------------------------
## 3. Eliminar filas no utiles

# Eliminar valores de loan_status no interesantes
data <- data %>%
  filter(loan_status %in% c('Late (16-30 days)', 'Late (31-120 days)', 'In Grace Period', 'Charged Off', 'Current')) 
  
#"in" Permite usar una operacion de filtrado apartir de un conjunto

# Grafica de la variable loan_status 
ggplot(data) +
    geom_histogram(aes(x = loan_status, fill = loan_status), stat= 'count')
  
## ---------------------------------------------------------------
## 4. Recodificar valores de clase "loan_status"

data <- data %>%
  mutate(loan_status = case_when(
    loan_status == 'Late (16-30 days)'  ~ 'Unpaid',
    loan_status == 'Late (31-120 days)' ~ 'Unpaid',
    loan_status == 'In Grace Period'    ~ 'Unpaid',
    loan_status == 'Charged Off'        ~ 'Unpaid',
    loan_status == 'Current'            ~ 'Paid'))

ggplot(data) +
  geom_histogram(aes(x = loan_status, fill = loan_status), stat = 'count')

#data <- data %>%
#  na.exclude() %>%
#  mutate(loan_status = as.factor(loan_status))

#summary(data$loan_status)

##Observamos equilbro entre Paid y Unpaid
table(data$loan_status)

## ---------------------------------------------------------------
## 5. Identificar columnos con alta correlacion

# Alta correlacion con la variable objetivo

data_num <- data %>%
  na.exclude() %>% #quitamos las filas que tenga NA
  mutate_if(is.character, as.factor) %>% 
  mutate_if(is.factor, as.numeric)
cor_target <- correlation_table(data_num, target='loan_status')

important_vars <- cor_target %>% 
  filter(abs(loan_status) >= 0.02) #Creamos un umbral
data <- data %>%
  select(one_of(important_vars$Variable))


# Alta correlacion entre si
data_num <- data %>%
  na.exclude() %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric)
rcorr_result <- rcorr(as.matrix(data_num))
cor_matrix <- as.tibble(rcorr_result$r, rownames = "variable")
corrplot(rcorr_result$r, type = "upper", order = "original", tl.col = "black", tl.srt = 45)

## Llamamos a la funcion varclus (Cluster)
v <- varclus(as.matrix(data_num), similarity="pearson") 
plot(v) #Diagramas que representar las agrupaciones 

# Podemos observar la correlacion entre si
View(rcorr_result$r) 

##Obtener cluster propiamente dicho
## Seleccionar variables 
#Umblar de coeficiente de pearson
#me quedo con una vareable de cada cluster 

groups <- cutree(v$hclust, 25) #25 es la cantidad de grupos, donde obtendre 25 variables

not_correlated_vars <- as.tibble(groups) %>%  #Pasa a dataframe 
  rownames_to_column() %>%  
  group_by(value) %>%  
  sample_n(1)

data <- data %>%
  select(one_of(not_correlated_vars$rowname))

View(data)
glimpse(data)

##Seleccionamos 30000
#data <- data[1:50000,]

#Guardamos el conjunyo de datos en un nuevo .csv
write.csv(data, file = "C:/Users/Javier Frere/Desktop/Entrega Sige/P_LoanStats_2017Q4.csv")


