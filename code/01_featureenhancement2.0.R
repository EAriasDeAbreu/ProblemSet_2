#0# Packages
install.packages(c("dplyr", "sf", "httr", "ggplot2", "rnaturalearth", "rnaturalearthdata", "glmnet", "caret"))
library(glmnet)
library(caret)
library(dplyr)
library(sf)
library(ggplot2)
library(rnaturalearth)
library(httr)
library(jsonlite)

#1# Import Datasets:

#1.1# A nivel de persona
train_data <- read.csv("/Users/lucia_mr/Dropbox/2. PEG/2024-2/BigData/ProblemSet2/Data/train_personas.csv")
test_data <- read.csv("/Users/lucia_mr/Dropbox/2. PEG/2024-2/BigData/ProblemSet2/Data/test_personas.csv")

#1.2# A nivel de hogar
train_hogares <- read.csv("/Users/lucia_mr/Dropbox/2. PEG/2024-2/BigData/ProblemSet2/Data/train_hogares.csv")
test_hogares <- read.csv("/Users/lucia_mr/Dropbox/2. PEG/2024-2/BigData/ProblemSet2/Data/test_hogares.csv")



if (!requireNamespace("googledrive", quietly = TRUE)) {
  install.packages("googledrive")
}
library(googledrive)

drive_auth()

# function to get creds (see here: https://www.kaggle.com/code/berent/r-api-for-kaggle-datase)
setup_kaggle_credentials <- function(username, key) {
  dir.create("~/.kaggle", showWarnings = FALSE)
  credentials <- list(username = username, key = key)
  write(toJSON(credentials, auto_unbox = TRUE), "~/.kaggle/kaggle.json")
  Sys.chmod("~/.kaggle/kaggle.json", mode = "0600")
}

# function to download data
download_competition_data <- function(competition) {
  url <- paste0("https://www.kaggle.com/api/v1/competitions/data/download-all/", competition)
  print(paste("Attempting to download from URL:", url))
  response <- GET(
    url,
    authenticate(Sys.getenv("KAGGLE_USERNAME"), Sys.getenv("KAGGLE_KEY")),
    write_disk(paste0(competition, ".zip"), overwrite = TRUE),
    progress()
  )
  
  print(paste("Response status code:", status_code(response)))
  if (status_code(response) == 200) {
    zip_file <- paste0(competition, ".zip")
    print(paste("Downloaded zip file:", zip_file))
    
    # list contents of the zip file
    zip_contents <- unzip(zip_file, list = TRUE)
    print("Contents of the zip file:")
    print(zip_contents)
    
    unzip(zip_file)
    file.remove(zip_file)
    print("Data extracted successfully.")
    
    # list files after extraction
    print("Files in the current directory after extraction:")
    print(list.files())
  } else {
    print(paste("Error downloading data. Status code:", status_code(response)))
    print(paste("Error message:", content(response, "text", encoding = "UTF-8")))
  }
}

# setup kaggle creds (for me: Edmundo :) )
setup_kaggle_credentials("", "YOUR_ACTUAL_KAGGLE_API_KEY")

# set env vars
Sys.setenv(KAGGLE_USERNAME = "")
Sys.setenv(KAGGLE_KEY = "")

# download data
competition_name <- "uniandes-bdml-2024-20-ps-2"
download_competition_data(competition_name)
print("Files in the directory after attempting to download:")
print(list.files())

# read data
if ("train_personas.csv" %in% list.files()) {
  train_data <- read.csv("train_personas.csv")
  print(head(train_data))
} else {
  print("train_personas.csv not found in the current directory.")
  print("Available files:")
  print(list.files())
}

if ("test_personas.csv" %in% list.files()) {
  test_data <- read.csv("test_personas.csv")
  print(head(test_data))
} else {
  print("test_personas.csv not found in the current directory.")
  print("Available files:")
  print(list.files())
}
if ("train_hogares.csv" %in% list.files()) {
  train_hogares <- read.csv("train_hogares.csv")
  print(head(train_hogares))
} else {
  print("train_hogares.csv not found in the current directory.")
  print("Available files:")
  print(list.files())
}
if ("test_hogares.csv" %in% list.files()) {
  test_hogares <- read.csv("test_hogares.csv")
  print(head(test_hogares))
} else {
  print("test_hogares.csv not found in the current directory.")
  print("Available files:")
  print(list.files())
}



#2# Data Cleaning:

#2.1# Keep only variables both in test and train
#common_columns <- intersect(names(train_data), names(test_data))
#train_data <- train_data[, common_columns]
#test_data <- train_data[, common_columns]

#2.2# Add the variable to predict: pobre

##Train
train_data <- train_data %>%
  left_join(train_hogares %>% select(id, Pobre, P5090, P5010, P5000, P5130), by = "id")

##For the last model
test_data <- test_data %>%
  left_join(test_hogares %>% select(id, P5090, P5010, P5000, P5130), by = "id")


#2.3# Transformaciones sobre train y test
datasets <- list("train_data", "test_data")

for (data in datasets) {
  assign(data, get(data) %>%
           group_by(id) %>%
           mutate(num_individuos = n()) %>%
           ungroup() %>%
           
           # Individuos menores de edad
           group_by(id) %>%
           mutate(num_individuos_menores_18 = sum(P6040 < 18, na.rm = TRUE)) %>%
           ungroup() %>%
           
           # Individuos mayores de 65
           group_by(id) %>%
           mutate(num_individuos_mayores_65 = sum(P6040 > 65, na.rm = TRUE)) %>%
           ungroup() %>%
           
           # Dummy mayores de 65
           group_by(id) %>%
           mutate(dummy_mayores_65 = as.integer(any(P6040 > 65))) %>%
           ungroup() %>%
           
           # ¿Es el jefe de hogar mujer?
           group_by(id) %>%
           mutate(jefe_mujer = as.integer(any(P6050 == 1 & P6020 == 2))) %>%
           ungroup() %>%
           
           
           # ¿Cotiza el jefe a pensión?
           group_by(id) %>%
           mutate(jefe_pension = as.integer(any(P6050 == 1 & P6090 == 2))) %>%
           ungroup() %>%
           mutate(jefe_pension = ifelse(is.na(jefe_pension), 0, jefe_pension)) %>%
           
           # ¿Es el jefe universitario?
           group_by(id) %>%
           mutate(jefe_uni = as.integer(any(P6050 == 1 & P6210 == 6))) %>%
           ungroup() %>%
           mutate(jefe_uni = ifelse(is.na(jefe_uni), 0, jefe_uni)) %>%
           
           # ¿Hay alguna persona en el hogar con educación universitaria o posgrado?
           group_by(id) %>%
           mutate(alguno_universidad = as.integer(any(P6210 == 6))) %>%
           ungroup() %>%
           
           # Eliminar menores de edad de la base
           #filter(P6040 >= 18) %>%
           
           # Porcentaje del hogar (+18) empleado
           group_by(id) %>%
           mutate(porcentaje_laborando = mean(P6240 == 1, na.rm = TRUE) * 100) %>%
           ungroup() %>%
           
           # Porcentaje del hogar (+18) empleado y Empleado en una empresa particular
           group_by(id) %>%
           mutate(share_empleado_privado = mean(P6240 == 1 & P6430 == 1, na.rm = TRUE) * 100) %>%
           ungroup() %>%
           
           # Porcentaje del hogar (+18) empleado y Empleado en una empresa pública
           group_by(id) %>%
           mutate(share_empleado_publico = mean(P6240 == 1 & P6430 == 2, na.rm = TRUE) * 100) %>%
           ungroup() %>%
           
           # Porcentaje del hogar (+18) empleado y cuenta propia
           group_by(id) %>%
           mutate(share_empleado_cuentapropia = mean(P6240 == 1 & P6430 == 4, na.rm = TRUE) * 100) %>%
           ungroup() %>%
           
           # ¿Alguien recibió alimentos como parte de pago?
           group_by(id) %>%
           mutate(alimentos = as.integer(any(P6590 == 1))) %>%
           ungroup() %>%
           mutate(alimentos = ifelse(is.na(alimentos), 0, alimentos)) %>%
           
           # ¿Alguien cotiza a pensión?
           group_by(id) %>%
           mutate(pension = as.integer(any(P6920 == 1))) %>%
           ungroup() %>%
           mutate(pension = ifelse(is.na(pension), 0, pension)) %>%
           
           # Horas trabajadas total del hogar (Semana)
           group_by(id) %>%
           mutate(horas_total_bruto = sum(P6800, na.rm = TRUE)) %>%
           ungroup() %>%
           
           # Horas trabajadas promedio del hogar (Semana)
           group_by(id) %>%
           mutate(horas_promedio_bruto = mean(P6800, na.rm = TRUE)) %>%
           ungroup() %>%
           mutate(horas_promedio_bruto = ifelse(is.na(horas_promedio_bruto), 0, horas_promedio_bruto)) %>%
           
           # ¿Alguien busca trabajar más horas? (No usaba, % missing del 94%)
           #group_by(id) %>%
           #mutate(subempleo_horas = as.integer(any(P7110 == 1))) %>%
           #ungroup() %>%
           #mutate(subempleo_horas = ifelse(is.na(subempleo_horas), 0, subempleo_horas)) %>%
           
           # ¿Alguien busca cambiar de trabajo?
           group_by(id) %>%
           mutate(subempleo_cap = as.integer(any(P7150 == 1))) %>%
           ungroup() %>%
           mutate(subempleo_cap = ifelse(is.na(subempleo_cap), 0, subempleo_cap)) %>%
           
           # ¿Alguien recibió pensión?
           group_by(id) %>%
           mutate(recibir_pension = as.integer(any(P7495 == 1))) %>%
           ungroup() %>%
           mutate(recibir_pension = ifelse(is.na(recibir_pension), 0, recibir_pension)) %>%
           
           # casa propia
           group_by(id) %>%
           mutate(casa_propia = as.integer(any(P5090 == 1))) %>%
           ungroup() %>%
           mutate(casa_propia = ifelse(is.na(casa_propia), 0, casa_propia)) %>%
           
           # PERSONAS POR CUARTO
           group_by(id) %>%
           mutate(cuartospersona = ifelse(!is.na(.data$P5000) & !is.na(.data$P5010) & .data$P5000 > 0, .data$P5010 / .data$P5000, NA_real_)) %>%
           ungroup() %>%
           mutate(cuartospersona = ifelse(is.na(cuartospersona), 0, cuartospersona)) %>%
           
           # ¿Recibió subsidios?
           group_by(id) %>%
           mutate(subsidio = as.integer(any(P7510s3 == 1))) %>%
           ungroup() %>%
           mutate(subsidio = ifelse(is.na(subsidio), 0, subsidio))
  )
}


#2.5# Drop duplicates at the household level

###Train
train_data <- train_data %>%
  distinct(id, .keep_all = TRUE)

###Test
test_data <- test_data %>%
  distinct(id, .keep_all = TRUE)


#2.6# Número de entradas faltantes en cada variable
missing_percentage <- sapply(train_data, function(x) {
  mean(is.na(x)) * 100
})

missing_df <- data.frame(
  Variable = names(missing_percentage),
  Missing_Percentage = missing_percentage
)

print(missing_df)
#---------------------------------------------------------------------------#
#---------------------------------------------------------------------------#
# Descriptive statistics

#1# Estrato por Pobres y no Pobres
install.packages("extrafont")
library(extrafont)
font_import()
loadfonts(device = "pdf")

theme_minimal(base_family = "serif")
#1.1# Train
my_plot <- ggplot(train_data, aes(x = factor(Estrato1), fill = factor(Pobre))) +
  geom_bar(position = "fill", color = "black", width = 0.7) +  # Stacked bars for proportion
  scale_fill_manual(values = c("#69b3a2", "#404080"), 
                    name = "Poverty Status", 
                    labels = c("Non-Poor", "Poor")) +  # Custom colors and legend
  labs(title = "",
       x = "Socioeconomic Stratum",
       y = "Proportion") + 
  theme_minimal() +  # Tema minimalista
  theme(
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold", color = "black"),  # Centered title, bold, black text
    axis.text = element_text(size = 14, color = "black"),  # Axis text in black
    axis.title = element_text(size = 16, color = "black", face = "italic"),  # Axis titles in italic and black
    panel.grid = element_blank(),  # Remove all grid lines
    legend.position = "bottom",  # Move legend below the graph
    legend.title = element_text(size = 14, face = "bold", color = "black"),  # Legend title in black
    legend.text = element_text(size = 12, color = "black")  # Legend text in black # Eliminar las líneas del grid
  )

my_plot
ggsave("/Users/lucia_mr/Dropbox/2. PEG/2024-2/BigData/ProblemSet2/Output/_train_poverty_distribution_by_stratum.pdf", 
       plot = my_plot, width = 8, height = 6, device = "pdf")


#-------------------------------------
#2# University Education per female household

#2.1#Train
# Resumir los datos utilizando la variable jefe_uni para female y male heads
summary_data <- train_data %>%
  summarize(
    jefe_mujer_avg = mean(jefe_mujer, na.rm = TRUE),
    alguno_universidad_female_jefe_avg = mean(jefe_uni[P6020 == 2], na.rm = TRUE),  # Promedio de jefe_uni para mujeres
    alguno_universidad_male_jefe_avg = mean(jefe_uni[P6020 == 1], na.rm = TRUE)  # Promedio de jefe_uni para hombres
  )

# Convertir los datos a formato largo para facilitar la creación del gráfico
summary_long <- data.frame(
  category = c("Female Head \n of Household",
               "University Education\n Female Head",
               "University Education\n Male Head"),
  average = c(summary_data$jefe_mujer_avg,
              summary_data$alguno_universidad_female_jefe_avg,
              summary_data$alguno_universidad_male_jefe_avg)
)

# Crear el gráfico con las etiquetas modificadas
femaletr <- ggplot(summary_long, aes(x = category, y = average)) +
  geom_bar(stat = "identity", fill = "#69b3a2", color = "black", width = 0.7) +  # Barras con color personalizado
  geom_vline(xintercept = 1.5, linetype = "dashed", color = "red", size = 1.2) +  # Línea vertical discontinua
  labs(title = "",
       x = "Category",  # Etiqueta del eje x en dos líneas
       y = "Proportion") +  # Etiquetas para el gráfico y ejes
  geom_text(aes(label = scales::percent(average, accuracy = 0.1)),  # Mostrar el porcentaje sobre cada barra
            vjust = -0.5, size = 5, color = "black") +  # Posicionar etiquetas justo arriba de cada barra
  ylim(0, 0.5) +  # Limitar el eje y de 0 a 0.5
  theme_minimal() +  # Tema minimalista
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Título centrado y en negrita
    axis.text = element_text(size = 12),  # Tamaño del texto en los ejes
    axis.title = element_text(size = 14),  # Tamaño del título de los ejes
    panel.grid = element_blank()  # Eliminar las líneas de cuadrícula
  )

femaletr

ggsave("/Users/lucia_mr/Dropbox/2. PEG/2024-2/BigData/ProblemSet2/Output/_train_female_educ.pdf", 
       plot = femaletr, width = 8, height = 6, device = "pdf")

#2.1# Test
summary_data <- test_data %>%
  summarize(
    jefe_mujer_avg = mean(jefe_mujer, na.rm = TRUE),
    alguno_universidad_female_jefe_avg = mean(jefe_uni[P6020 == 2], na.rm = TRUE),  # Promedio de jefe_uni para mujeres
    alguno_universidad_male_jefe_avg = mean(jefe_uni[P6020 == 1], na.rm = TRUE)  # Promedio de jefe_uni para hombres
  )

# Convertir los datos a formato largo para facilitar la creación del gráfico
summary_long <- data.frame(
  category = c("Female Head \n of Household",
               "University Education\n Female Head",
               "University Education\n Male Head"),
  average = c(summary_data$jefe_mujer_avg,
              summary_data$alguno_universidad_female_jefe_avg,
              summary_data$alguno_universidad_male_jefe_avg)
)

# Crear el gráfico con las etiquetas modificadas
femaletest <- ggplot(summary_long, aes(x = category, y = average)) +
  geom_bar(stat = "identity", fill = "#69b3a2", color = "black", width = 0.7) +  # Barras con color personalizado
  geom_vline(xintercept = 1.5, linetype = "dashed", color = "red", size = 1.2) +  # Línea vertical discontinua
  labs(title = "",
       x = "Category",  # Etiqueta del eje x en dos líneas
       y = "Proportion") +  # Etiquetas para el gráfico y ejes
  geom_text(aes(label = scales::percent(average, accuracy = 0.1)),  # Mostrar el porcentaje sobre cada barra
            vjust = -0.5, size = 5, color = "black") +  # Posicionar etiquetas justo arriba de cada barra
  ylim(0, 0.5) +  # Limitar el eje y de 0 a 0.5
  theme_minimal() +  # Tema minimalista
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Título centrado y en negrita
    axis.text = element_text(size = 12),  # Tamaño del texto en los ejes
    axis.title = element_text(size = 14),  # Tamaño del título de los ejes
    panel.grid = element_blank()  # Eliminar las líneas de cuadrícula
  )
femaletest
ggsave("/Users/lucia_mr/Dropbox/2. PEG/2024-2/BigData/ProblemSet2/Output/_test_female_educ.pdf", 
       plot = femaletest, width = 8, height = 6, device = "pdf")


#3# Table of descriptive stats:
# Load necessary libraries
library(dplyr)
library(tidyr)
library(knitr)
library(broom)

# Function to calculate the percentage of non-missing values
non_missing_percent <- function(x) {
  100 * sum(!is.na(x)) / length(x)
}

# Transform income variables to millions
train_data <- train_data %>%
  mutate(
    pension = pension / 1e6,
    subsidio = subsidio / 1e6
  )

test_data <- test_data %>%
  mutate(
    pension = pension / 1e6,
    subsidio = subsidio / 1e6
  )

# Variables to include in the summary (without ingreso_total_bruto, ingreso_promedio_bruto, subempleo_horas, and subempleo_cap)
variables <- c("num_individuos", "num_individuos_menores_18", "num_individuos_mayores_65",
               "dummy_mayores_65", "jefe_mujer", "alguno_universidad", "porcentaje_laborando",
               "share_empleado_privado", "share_empleado_publico", "share_empleado_cuentapropia",
               "alimentos", "horas_total_bruto", "horas_promedio_bruto", "pension", "subsidio")

# Calculate the mean, % non-missing, and standard deviation for train and test datasets
train_summary <- data.frame(
  "Variable" = variables,
  "Non-missing (%) (Train)" = sapply(train_data[, variables], non_missing_percent),
  "Mean (Train)" = round(sapply(train_data[, variables], mean, na.rm = TRUE), 2),
  "Std Dev (Train)" = round(sapply(train_data[, variables], sd, na.rm = TRUE), 2)
)

test_summary <- data.frame(
  "Non-missing (%) (Test)" = sapply(test_data[, variables], non_missing_percent),
  "Mean (Test)" = round(sapply(test_data[, variables], mean, na.rm = TRUE), 2),
  "Std Dev (Test)" = round(sapply(test_data[, variables], sd, na.rm = TRUE), 2)
)

# Perform t-tests for significant differences between train and test datasets
t_tests <- sapply(variables, function(var) {
  t_test <- t.test(train_data[[var]], test_data[[var]], na.rm = TRUE)
  return(t_test$p.value)
})

t_tests_df <- data.frame("T-test p-value" = round(t_tests, 3))

# Combine the results into a single summary table
summary_table <- cbind(train_summary, test_summary, t_tests_df)

# Export the table to LaTeX format using kable in markdown style
latex_table <- kable(summary_table, format = "latex", booktabs = TRUE, align = "c", col.names = c("Variable",
                                                                                                  "Non-missing (%) (Train)", "Mean (Train)", "Std Dev (Train)",
                                                                                                  "Non-missing (%) (Test)", "Mean (Test)", "Std Dev (Test)",
                                                                                                  "T-test p-value"))



# Export the table to a .tex file
writeLines(latex_table, "/Users/lucia_mr/Dropbox/2. PEG/2024-2/BigData/ProblemSet2/Output/summary_table.tex")

