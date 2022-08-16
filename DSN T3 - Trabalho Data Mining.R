#Alumni Coppead - DSN T3
#Data Mining
#Grupo: André Rohloff , Camila Campos , Gabriel Tambelli , Matheus Cadorini e Rafaella Gregorio
#######################################

## Limpeza da memória
rm(list = ls())
gc(full = TRUE)

### Carregamento de bibliotecas
library(ggplot2)
library(gridExtra)
library(AER)
library(GGally)
library(caret)
library(UBL)
library(nnet)
library(doParallel)
library(kernlab)

## Definição de diretório da base de dados
path<-"C:/Users/cador/OneDrive/Área de Trabalho/Documentos COPPEAD/Data Mining"
setwd(path)

## Leitura da base de dados
bd_HR <- data.frame(read.csv("HR Employee Attrition.csv"))

### ANÁLISE EXPLORATÓRIA DA BASE DE DADOS

## verificação de missing values
paste("Número de missing values na base de dados: ", sum(is.na(bd_HR)))

## dimensões da base
paste("Dimensões da base de dados: ", dim(bd_HR)[1], "x", dim(bd_HR)[2])

## tipos das variáveis
paste("Tipos das variáveis: ")
str(bd_HR)

## convertendo variáveis caracterE para factor e guardando quais colunas são factor
aux_factor <- c()
for(i in 1:ncol(bd_HR)){
  if(is.character(bd_HR[1,i]) == TRUE) {
    bd_HR[,i] <- factor(bd_HR[,i])
    aux_factor <- append(i,aux_factor)
  }
}
aux_factor <- sort(aux_factor,decreasing = FALSE)

## tipos das variáveis após modificação para factor
paste("Tipos das variáveis c/ factor: ")
str(bd_HR)

## verificando nomes das colunas e renomeando a primeira
paste("Nomes das colunas ajustado: ")
colnames(bd_HR)[1] <- "Age"
colnames(bd_HR)

# loop para montar os gráficos de barras para todas as variáveis categóricas
for (i in aux_factor){
  cont <- data.frame(table(bd_HR[,i]))
  assign(paste0("plot_", colnames(bd_HR)[i]),ggplot(data = cont)+
                              geom_col(aes(x = Var1, y = Freq))+
                              geom_label(aes(x = Var1, y = Freq + 2, label = Freq)) +
                              labs(title= paste("Freq. ", colnames(bd_HR)[i]), x = colnames(bd_HR)[i], y = "Ocorrências")+
                              theme(plot.title=element_text(size=12, colour="grey30", hjust = 0.5, face = "bold"),
                                    panel.grid.major.x = element_blank(),  
                                    axis.text.x=element_text(angle=90, size=8),
                                    axis.title.x=element_text(face="bold", size=8),
                                    axis.title.y=element_text(face="bold", size=8),
                                    legend.title=element_text(face="bold"),
                                    legend.position = "bottom"))
  print(eval(parse(text=paste0("plot_", colnames(bd_HR)[i]))))
}

## a variável categórica "Over18" possui apenas uma classe e pode ser excluída do dataset
bd_HR <- bd_HR[,!colnames(bd_HR) %in% c("Over18")]

## criação de tabela com estatísticas descritivas

## seleção apenas das colunas numéricas do dataset
bd_HR_num <- bd_HR[,-aux_factor]

## inicialização da matriz
tabela_estat <- matrix(NA,5,ncol(bd_HR_num))
colnames(tabela_estat) <- colnames(bd_HR_num)
row.names(tabela_estat) <- c("Mínimo", "Máximo", "Média", "Mediana", "Desvio Padrão")

## cálculo das estatísticas
for (i in 1:ncol(tabela_estat)){
tabela_estat[1,i] <- min(bd_HR_num[,i])
tabela_estat[2,i] <- max(bd_HR_num[,i])
tabela_estat[3,i] <- mean(bd_HR_num[,i])
tabela_estat[4,i] <- median(bd_HR_num[,i])
tabela_estat[5,i] <- sd(bd_HR_num[,i])
}

## as colunas EmployeeCount e StandardHours têm desvpad nulo, portanto também podem ser excluídas
aux_excl <- colnames(tabela_estat)[tabela_estat[5, ] == 0]
bd_HR <- bd_HR[,!colnames(bd_HR) %in% aux_excl]

## podemos ver pela análise abaixo que claramente o dataset é desbalanceado
table(bd_HR$Attrition)

### SEPARACAO EM TREINO E TESTE

## abaixo, vamos separar o dataset em treino e teste mantendo a proporção das classes
## da variável Attrition
pos_treino <- createDataPartition(y = bd_HR$Attrition, p = 0.8, list = FALSE)

## Dado de treino
data_treino <- bd_HR[pos_treino, ]

## Dado de teste
data_teste <- bd_HR[-pos_treino, ]

### CRIANDO DATASET BALANCEADO PELA TECNICA DE OVERSAMPLING

table(data_treino$Attrition)

## iniciando o dataset com oversampling
data_treino_over <- NULL

## contagems das classes
max_class <- max(table(data_treino$Attrition))

## Loop nas classes
for(i in levels(data_treino$Attrition)){
  ## dataset apenas com a classe i
  data_i <- data_treino[data_treino$Attrition == i, ]
  
  ## sample com a quantidade máxima
  pos <- sample(x = 1:nrow(data_i), size = max_class, 
                replace = ifelse(nrow(data_i) == max_class, FALSE, TRUE))
  data_i <- data_i[pos, ]
  
  ## adiciona no dataset 
  data_treino_over <- rbind(data_treino_over, data_i)
  
}

## verificando o balanceamento do dataset após o oversampling
table(data_treino_over$Attrition)

## DATASET DE TREINO SEM TRATAMENTO DE DESBALANCEAMENTO

## MODELO REGRESSÃO LOGÍSTICA

## definicao dos parametros a serem validados
L2_ub <- 10**seq(from = -5, to = 0, by = 0.1)
decay_ub <- expand.grid(decay = L2_ub)

## quantidade de parametros
print("Quantidade de modelos RLog desbalanceado:")
nrow(decay_ub)

## descricao simbolica
frl <- Attrition ~.

## treino da regressão logística com validacao dos parametros - Execucao em paralelo
system.time({
  ## cluster de execucao paralela
  cl <- makePSOCKcluster(detectCores())
  registerDoParallel(cl)
  
  ## validacao do caret
  reglog_val_ub <- train(frl, data = data_treino, method = "multinom", metric = "Kappa", cache = 2000,
                   trControl = trainControl(method = "repeatedcv",
                                            number = 10, repeats = 3),
                   tuneGrid = decay_ub)
  
  ## fecha o cluster paralelo
  stopCluster(cl)
  
})

## melhor parametro decay
reglog_val_ub$bestTune

## grafico do erro de validacao
ggplot(reglog_val_ub$results) + 
  geom_line(aes(x = decay, y = Kappa)) +
  xlab("Decay") + ylab("Kappa (Validacao)") + 
  theme_bw()

## treino da melhor regressão logística para o dataset desbalanceado
reglog_model_ub <- multinom(formula = frl, data = data_treino, 
                  maxit = 1000, trace = FALSE, decay = reglog_val_ub$bestTune)

## teste do modelo treinado
reglog_teste_ub <- predict(reglog_model_ub, data_teste)

## matriz de confusão do modelo de regressão logística com desbalanceamento
cm_reglog_ub <- confusionMatrix(reglog_teste_ub,
                                 data_teste$Attrition, positive='Yes')
print(cm_reglog_ub)
cm_reglog_ub$byClass

## data frame com parametros comparativos dos modelos
param_modelos <- NULL

## armazenamento resultados reglog desbalanceado
param_modelos <- rbind(param_modelos, 
                   c(Model = "reglog_ub", cm_reglog_ub$overall[1], cm_reglog_ub$overall[2], 
                      cm_reglog_ub$byClass[1], cm_reglog_ub$byClass[2], cm_reglog_ub$byClass[7]))

## MODELO RANDOM FOREST

## definicao dos parametros a serem validados
mtry_ub <- seq(from = 7, to = 15, by = 1) ## Qtde de variaveis 
grid_rf_ub <- expand.grid(mtry = mtry_ub) 
head(grid_rf_ub)

## quantidade de parametros
print("Quantidade de modelos RF desbalanceado:")
nrow(grid_rf_ub)

## treino do RF com validacao de mtry - Execucao em paralelo
system.time({
  ## cluster de execucao paralela
  cl <- makePSOCKcluster(detectCores())
  registerDoParallel(cl)
  
  ## validacao do caret
  rf_val_ub <- train(frl, data = data_treino, ntree = 2000, method = "rf", metric = "Kappa",
                        trControl = trainControl(method = "repeatedcv",
                                                 number = 10, repeats = 3),
                        tuneGrid = grid_rf_ub)
  
  ## fecha o cluster paralelo
  stopCluster(cl)
  
})

## melhor parametro mtry
cat("Melhor mtry desbalanceado:", rf_val_ub$bestTune$mtry, "\n")

## grafico do erro de validacao
ggplot(rf_val_ub$results) + 
  geom_line(aes(x = mtry, y = Kappa)) +
  xlab("mtry") + ylab("Kappa (Validacao)") + 
  theme_bw()

## treino da melhor random forest para o dataset desbalanceado
rf_model_ub <- randomForest(formula = frl, data = data_treino, ntree = 2000, 
                            mtry = rf_val_ub$bestTune$mtry, importance = TRUE)

## teste do modelo treinado
rf_teste_ub <- predict(rf_model_ub, data_teste)

## matriz de confusão do modelo de random forest com desbalanceamento
cm_rf_ub <- confusionMatrix(rf_teste_ub,
                                data_teste$Attrition, positive='Yes')
print(cm_rf_ub)
cm_rf_ub$byClass

## armazenamento resultados rf desbalanceado
param_modelos <- rbind(param_modelos, 
                       c(Model = "rf_ub", cm_rf_ub$overall[1], cm_rf_ub$overall[2], 
                         cm_rf_ub$byClass[1], cm_rf_ub$byClass[2], cm_rf_ub$byClass[7]))


## MODELO SVM RADIAL

## definicao dos parametros a serem validados
sigma_ub <- 10**seq(from = -4, to = -2, by = 0.1)
C_ub     <- 10**seq(from = 0, to = 2, by = 0.1)
grid_svm_ub <- expand.grid(sigma = sigma_ub, C = C_ub)

## quantidade de parametros
print("Quantidade de modelos SVM desbalanceado:")
nrow(grid_svm_ub)

## treino do SVM com validacao dos parametros
system.time({
  cl <- makePSOCKcluster(detectCores())
  registerDoParallel(cl)
  
  ## validacao do caret
  svm_val_ub <- train(frl, data = data_treino, method = "svmRadial", metric = "Kappa", cache = 2000,
                   trControl = trainControl(method = "repeatedcv",
                                            number = 10, repeats = 3),
                   tuneGrid = grid_svm_ub)
  
  ## Fecha o cluster paralelo
  stopCluster(cl)
  
})

## melhores parametros svm
svm_val_ub$bestTune

## treino da melhor random forest para o dataset desbalanceado
svm_model_ub <- ksvm(x = frl, data = data_treino, type = "C-svc", kernel = "rbfdot",
                     kpar = list(sigma = svm_val_ub$bestTune$sigma), cache = 2000,
                     C = svm_val_ub$bestTune$C)

## teste do modelo treinado
svm_teste_ub <- predict(svm_model_ub, data_teste)

## matriz de confusão do modelo svm com desbalanceamento
cm_svm_ub <- confusionMatrix(svm_teste_ub,
                            data_teste$Attrition, positive='Yes')
print(cm_svm_ub)
cm_svm_ub$byClass

## armazenamento resultados svm radial desbalanceado
param_modelos <- rbind(param_modelos, 
                       c(Model = "svm_ub", cm_svm_ub$overall[1], cm_svm_ub$overall[2], 
                         cm_svm_ub$byClass[1], cm_svm_ub$byClass[2], cm_svm_ub$byClass[7]))

## ------------------------------------------------------------------------------------------
## DATASET DE TREINO COM TRATAMENTO DE DESBALANCEAMENTO

## para os datasets balanceados, a métrica de decisão do treino do caret será mudada
## de Kappa para Accuracy, justamente por conta do balanceamento

## MODELO REGRESSÃO LOGÍSTICA

## definicao dos parametros a serem validados
L2_b <- 10**seq(from = -3, to = 0, by = 0.1)
decay_b <- expand.grid(decay = L2_b)

## quantidade de parametros
print("Quantidade de modelos RLog balanceado:")
nrow(decay_b)

## treino da regressão logística com validacao dos parametros - Execucao em paralelo
system.time({
  ## cluster de execucao paralela
  cl <- makePSOCKcluster(detectCores())
  registerDoParallel(cl)
  
  ## validacao do caret
  reglog_val_b <- train(frl, data = data_treino_over, method = "multinom", metric = "Accuracy", cache = 2000,
                         trControl = trainControl(method = "repeatedcv",
                                                  number = 10, repeats = 3),
                         tuneGrid = decay_b)
  
  ## fecha o cluster paralelo
  stopCluster(cl)
  
})

## melhor parametro decay
reglog_val_b$bestTune

## grafico do erro de validacao
ggplot(reglog_val_b$results) + 
  geom_line(aes(x = decay, y = Kappa)) +
  xlab("Decay") + ylab("Kappa (Validacao)") + 
  theme_bw()

## treino da melhor regressão logística para o dataset balanceado
reglog_model_b <- multinom(formula = frl, data = data_treino_over, 
                            maxit = 1000, trace = FALSE, decay = reglog_val_b$bestTune)

## teste do modelo treinado
reglog_teste_b <- predict(reglog_model_b, data_teste)

## matriz de confusão do modelo de regressão logística balanceado
cm_reglog_b <- confusionMatrix(reglog_teste_b,
                                data_teste$Attrition, positive='Yes')
print(cm_reglog_b)
cm_reglog_b$byClass

## armazenamento resultados reglog balanceado
param_modelos <- rbind(param_modelos, 
                       c(Model = "reglog_b", cm_reglog_b$overall[1], cm_reglog_b$overall[2], 
                         cm_reglog_b$byClass[1], cm_reglog_b$byClass[2], cm_reglog_b$byClass[7]))

## MODELO RANDOM FOREST

## definicao dos parametros a serem validados
mtry_b <- seq(from = 2, to = 6, by = 1)
grid_rf_b <- expand.grid(mtry = mtry_b) 
head(grid_rf_b)

## quantidade de parametros
print("Quantidade de modelos RF balanceado:")
nrow(grid_rf_b)

## treino do RF com validacao de mtry - Execucao em paralelo
system.time({
  ## cluster de execucao paralela
  cl <- makePSOCKcluster(detectCores())
  registerDoParallel(cl)
  
  ## validacao do caret
  rf_val_b <- train(frl, data = data_treino_over, ntree = 2000, method = "rf", metric = "Accuracy",
                     trControl = trainControl(method = "repeatedcv",
                                              number = 10, repeats = 3),
                     tuneGrid = grid_rf_b)
  
  ## fecha o cluster paralelo
  stopCluster(cl)
  
})

## melhor parametro mtry balanceado
cat("Melhor mtry balanceado:", rf_val_b$bestTune$mtry, "\n")

## grafico do erro de validacao
ggplot(rf_val_b$results) + 
  geom_line(aes(x = mtry, y = Kappa)) +
  xlab("mtry") + ylab("Kappa (Validacao)") + 
  theme_bw()

## treino da melhor random forest para o dataset desbalanceado
rf_model_b <- randomForest(formula = frl, data = data_treino_over, ntree = 2000, 
                            mtry = rf_val_b$bestTune$mtry, importance = TRUE)

## plot das importancias das variaveis - Over Time é a mais importante para a queda de acurácia na permutacao
varImpPlot(rf_model_b, type = 1, main = "Attrition Balanced Random Forest")

## teste do modelo treinado
rf_teste_b <- predict(rf_model_b, data_teste)

## matriz de confusão do modelo de random forest sem desbalanceamento
cm_rf_b <- confusionMatrix(rf_teste_b,
                            data_teste$Attrition, positive='Yes')
print(cm_rf_b)
cm_rf_b$byClass

## armazenamento resultados rf balanceado
param_modelos <- rbind(param_modelos, 
                       c(Model = "rf_b", cm_rf_b$overall[1], cm_rf_b$overall[2], 
                         cm_rf_b$byClass[1], cm_rf_b$byClass[2], cm_rf_b$byClass[7]))


## MODELO SVM RADIAL

## definicao dos parametros a serem validados
sigma_b <- 10**seq(from = 4, to = 10, by = 0.2)
C_b <- 10**seq(from = -1, to = 2, by = 0.2)
grid_svm_b <- expand.grid(sigma = sigma_b, C = C_b)

## quantidade de parametros
print("Quantidade de modelos SVM balanceado:")
nrow(grid_svm_b)

## treino do SVM com validacao dos parametros
system.time({
  cl <- makePSOCKcluster(detectCores())
  registerDoParallel(cl)
  
  ## validacao do caret
  svm_val_b <- train(frl, data = data_treino_over, method = "svmRadial", metric = "Accuracy", cache = 2000,
                      trControl = trainControl(method = "repeatedcv",
                                               number = 10, repeats = 3),
                      tuneGrid = grid_svm_b)
  
  ## Fecha o cluster paralelo
  stopCluster(cl)
  
})

## melhores parametros svm
svm_val_b$bestTune

## treino do melhor svm para o dataset balanceado
svm_model_b <- ksvm(x = frl, data = data_treino_over, type = "C-svc", kernel = "rbfdot",
                     kpar = list(sigma = svm_val_b$bestTune$sigma), cache = 2000,
                     C = svm_val_b$bestTune$C)

## teste do modelo treinado
svm_teste_b <- predict(svm_model_b, data_teste)

## matriz de confusão do modelo svm sem desbalanceamento
cm_svm_b <- confusionMatrix(svm_teste_b,
                             data_teste$Attrition, positive='Yes')
print(cm_svm_b)
cm_svm_b$byClass

## armazenamento resultados svm radial balanceado
param_modelos <- rbind(param_modelos, 
                       c(Model = "svm_b", cm_svm_b$overall[1], cm_svm_b$overall[2], 
                         cm_svm_b$byClass[1], cm_svm_b$byClass[2], cm_svm_b$byClass[7]))

## como o modelo svm radial perdeu o poder de predicao de attrition para o dataset balanceado
## vamos testar uma outra alternativa de svm, dessa vez com kernel polinomial

## MODELO SVM POLINOMIAL

## definicao dos parametros a serem validados
degree_poly_b <- seq(from = 3, to = 6, by = 1)
scale_poly_b <- 10**seq(from = 0, to = 3, by = 1)
C_poly_b <- 10**seq(from = -3, to = 0, by = 0.1)
grid_svm_poly_b <- expand.grid(degree = degree_poly_b, scale = scale_poly_b,  C = C_poly_b)

## quantidade de parametros
print("Quantidade de modelos SVM balanceado:")
nrow(grid_svm_poly_b)

## treino do SVM com validacao dos parametros
system.time({
  cl <- makePSOCKcluster(detectCores())
  registerDoParallel(cl)
  
  ## validacao do caret
  svm_poly_val_b <- train(frl, data = data_treino_over, method = "svmPoly", metric = "Accuracy", cache = 2000,
                     trControl = trainControl(method = "repeatedcv",
                                              number = 10, repeats = 3),
                     tuneGrid = grid_svm_poly_b)
  
  ## Fecha o cluster paralelo
  stopCluster(cl)
  
})

## melhores parametros svm polinomial
svm_poly_val_b$bestTune

## treino do melhor svm para o dataset balanceado
svm_poly_model_b <- ksvm(x = frl, data = data_treino_over, type = "C-svc", kernel = "polydot",
                    kpar = list(degree = svm_poly_val_b$bestTune$degree, scale = svm_poly_val_b$bestTune$scale), cache = 2000,
                    C = svm_poly_val_b$bestTune$C)

## teste do modelo treinado
svm_poly_teste_b <- predict(svm_poly_model_b, data_teste)

## matriz de confusão do modelo svm sem desbalanceamento
cm_svm_poly_b <- confusionMatrix(svm_poly_teste_b,
                            data_teste$Attrition, positive='Yes')
print(cm_svm_poly_b)
cm_svm_poly_b$byClass

## armazenamento resultados svm polinomial balanceado
param_modelos <- rbind(param_modelos, 
                       c(Model = "svm_poly_b", cm_svm_poly_b$overall[1], cm_svm_poly_b$overall[2], 
                         cm_svm_poly_b$byClass[1], cm_svm_poly_b$byClass[2], cm_svm_poly_b$byClass[7]))

## ANÁLISE COMPARATIVA ENTRE MODELOS
param_modelos

## o modelo de svm com kernel polinomial para o dataset balanceado voltou a ter poder de 
## predicao para Attrition, mas mesmo assim, quando olhamos para as métricas Kappa ou F1
## (já que o dataset de teste continua desbalanceado por natureza), percebemos que o
## modelo de regressão logística para o dataset de treino desbalanceado foi o que
## apresentou melhor desempenho e, portanto, seria o mais adequado para este problema
## (o resultado com o dataset desbalanceado ter sido melhor que com o dataset balanceado
## foi um surpresa para o grupo, bem como o modelo de regressão logística ter dado melhor
## que o de random forest e svm).