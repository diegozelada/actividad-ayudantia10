Ayudantia 10 Metodo Bayesiano
================

## Actividad Ayudantia 10

Para esta ayudantia tendran que realizar el metodo de analisis bayesiano
para el data set de tarjetas de credito (dentro del bloc de notas podran
encontrar el link que la explicacion de las variables del data set). El
objetivo para ese data set es determinar si la persona fallara o no en
el pago de su credito a partir de las variables del dataset

## Cargamos Librerias

``` r
library(tidyverse)
library(e1071)
library(caret)
library(rstan)
library(rstanarm)
library(titanic)
```

``` r
#rstan_options(auto_write=TRUE)
# Run on multiple cores
#options(mc.cores = parallel::detectCores())
```

## Cargamos los datos con los que vamos a trabajar

``` r
titanictrain <- titanic::titanic_train %>%  as.data.frame()
titanictest <- titanic::titanic_test %>% as.data.frame()

titanic <- titanictrain
```

``` r
glimpse(titanic)
```

    ## Rows: 891
    ## Columns: 12
    ## $ PassengerId <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,~
    ## $ Survived    <int> 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1~
    ## $ Pclass      <int> 3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3~
    ## $ Name        <chr> "Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley (Fl~
    ## $ Sex         <chr> "male", "female", "female", "female", "male", "male", "mal~
    ## $ Age         <dbl> 22, 38, 26, 35, 35, NA, 54, 2, 27, 14, 4, 58, 20, 39, 14, ~
    ## $ SibSp       <int> 1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0, 0, 4, 0, 1, 0~
    ## $ Parch       <int> 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 5, 0, 0, 1, 0, 0, 0~
    ## $ Ticket      <chr> "A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "37~
    ## $ Fare        <dbl> 7.2500, 71.2833, 7.9250, 53.1000, 8.0500, 8.4583, 51.8625,~
    ## $ Cabin       <chr> "", "C85", "", "C123", "", "", "E46", "", "", "", "G6", "C~
    ## $ Embarked    <chr> "S", "C", "S", "S", "S", "Q", "S", "S", "S", "C", "S", "S"~

## Pre Procesamiento

``` r
titanic$class <- str_extract(titanictrain$Pclass, "[0-9]")
titanic$SexCode <- (titanic$Sex == "female") %>% as.numeric()

titanic <- titanic[c(4,3,5:14,2)]

str(titanic)
```

    ## 'data.frame':    891 obs. of  13 variables:
    ##  $ Name    : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
    ##  $ Pclass  : int  3 1 3 1 3 3 1 3 3 2 ...
    ##  $ Sex     : chr  "male" "female" "female" "female" ...
    ##  $ Age     : num  22 38 26 35 35 NA 54 2 27 14 ...
    ##  $ SibSp   : int  1 1 0 1 0 0 0 3 0 1 ...
    ##  $ Parch   : int  0 0 0 0 0 0 0 1 2 0 ...
    ##  $ Ticket  : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
    ##  $ Fare    : num  7.25 71.28 7.92 53.1 8.05 ...
    ##  $ Cabin   : chr  "" "C85" "" "C123" ...
    ##  $ Embarked: chr  "S" "C" "S" "S" ...
    ##  $ class   : chr  "3" "1" "3" "1" ...
    ##  $ SexCode : num  0 1 1 1 0 0 0 0 1 1 ...
    ##  $ Survived: int  0 1 1 1 0 0 0 0 1 1 ...

``` r
titanic$Sex <- NULL
titanic$Ticket <- NULL
titanic$Cabin <- NULL
titanic$SibSp <- NULL
titanic$Parch <- NULL
titanic$Fare <- NULL
titanic$Embarked <- NULL
titanic$Pclass <- NULL

str(titanic)
```

    ## 'data.frame':    891 obs. of  5 variables:
    ##  $ Name    : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
    ##  $ Age     : num  22 38 26 35 35 NA 54 2 27 14 ...
    ##  $ class   : chr  "3" "1" "3" "1" ...
    ##  $ SexCode : num  0 1 1 1 0 0 0 0 1 1 ...
    ##  $ Survived: int  0 1 1 1 0 0 0 0 1 1 ...

``` r
titanictest$class <- str_extract(titanictest$Pclass, "[0-9]")
titanictest$SexCode <- (titanictest$Sex == "female") %>% as.numeric()

titanictest <- titanictest[c(3,4,5:13,2)]

str(titanictest)
```

    ## 'data.frame':    418 obs. of  12 variables:
    ##  $ Name    : chr  "Kelly, Mr. James" "Wilkes, Mrs. James (Ellen Needs)" "Myles, Mr. Thomas Francis" "Wirz, Mr. Albert" ...
    ##  $ Sex     : chr  "male" "female" "male" "male" ...
    ##  $ Age     : num  34.5 47 62 27 22 14 30 26 18 21 ...
    ##  $ SibSp   : int  0 1 0 0 1 0 0 1 0 2 ...
    ##  $ Parch   : int  0 0 0 0 1 0 0 1 0 0 ...
    ##  $ Ticket  : chr  "330911" "363272" "240276" "315154" ...
    ##  $ Fare    : num  7.83 7 9.69 8.66 12.29 ...
    ##  $ Cabin   : chr  "" "" "" "" ...
    ##  $ Embarked: chr  "Q" "S" "Q" "S" ...
    ##  $ class   : chr  "3" "3" "2" "3" ...
    ##  $ SexCode : num  0 1 0 0 1 0 1 0 1 0 ...
    ##  $ Pclass  : int  3 3 2 3 3 3 3 2 3 3 ...

``` r
titanictest$Sex <- NULL
titanictest$Ticket <- NULL
titanictest$Cabin <- NULL
titanictest$SibSp <- NULL
titanictest$Parch <- NULL
titanictest$Fare <- NULL
titanictest$Embarked <- NULL
titanictest$Pclass <- NULL

str(titanictest)
```

    ## 'data.frame':    418 obs. of  4 variables:
    ##  $ Name   : chr  "Kelly, Mr. James" "Wilkes, Mrs. James (Ellen Needs)" "Myles, Mr. Thomas Francis" "Wirz, Mr. Albert" ...
    ##  $ Age    : num  34.5 47 62 27 22 14 30 26 18 21 ...
    ##  $ class  : chr  "3" "3" "2" "3" ...
    ##  $ SexCode: num  0 1 0 0 1 0 1 0 1 0 ...

## Metodo Bayesiano

``` r
library(e1071)

TitanicLinear <- stan_glm(Survived ~ Age + SexCode + as.factor(class), 
                          data = titanic, family = gaussian)
```

    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 0 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 1: Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 1: Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 1: Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 1: Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 2000 / 2000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 0.237 seconds (Warm-up)
    ## Chain 1:                0.4 seconds (Sampling)
    ## Chain 1:                0.637 seconds (Total)
    ## Chain 1: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
    ## Chain 2: 
    ## Chain 2: Gradient evaluation took 0 seconds
    ## Chain 2: 1000 transitions using 10 leapfrog steps per transition would take 0 seconds.
    ## Chain 2: Adjust your expectations accordingly!
    ## Chain 2: 
    ## Chain 2: 
    ## Chain 2: Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 2: Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 2: Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 2: Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 2: Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 2: Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 2: Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 2: Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 2: Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 2: Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 2: Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 2: Iteration: 2000 / 2000 [100%]  (Sampling)
    ## Chain 2: 
    ## Chain 2:  Elapsed Time: 0.195 seconds (Warm-up)
    ## Chain 2:                0.461 seconds (Sampling)
    ## Chain 2:                0.656 seconds (Total)
    ## Chain 2: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
    ## Chain 3: 
    ## Chain 3: Gradient evaluation took 0 seconds
    ## Chain 3: 1000 transitions using 10 leapfrog steps per transition would take 0 seconds.
    ## Chain 3: Adjust your expectations accordingly!
    ## Chain 3: 
    ## Chain 3: 
    ## Chain 3: Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 3: Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 3: Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 3: Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 3: Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 3: Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 3: Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 3: Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 3: Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 3: Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 3: Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 3: Iteration: 2000 / 2000 [100%]  (Sampling)
    ## Chain 3: 
    ## Chain 3:  Elapsed Time: 0.322 seconds (Warm-up)
    ## Chain 3:                0.42 seconds (Sampling)
    ## Chain 3:                0.742 seconds (Total)
    ## Chain 3: 
    ## 
    ## SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
    ## Chain 4: 
    ## Chain 4: Gradient evaluation took 0 seconds
    ## Chain 4: 1000 transitions using 10 leapfrog steps per transition would take 0 seconds.
    ## Chain 4: Adjust your expectations accordingly!
    ## Chain 4: 
    ## Chain 4: 
    ## Chain 4: Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 4: Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 4: Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 4: Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 4: Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 4: Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 4: Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 4: Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 4: Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 4: Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 4: Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 4: Iteration: 2000 / 2000 [100%]  (Sampling)
    ## Chain 4: 
    ## Chain 4:  Elapsed Time: 0.19 seconds (Warm-up)
    ## Chain 4:                0.322 seconds (Sampling)
    ## Chain 4:                0.512 seconds (Total)
    ## Chain 4:

``` r
model_nb <- naiveBayes(Survived ~ Age + SexCode + as.factor(class), titanic, laplace=1)
```

``` r
#pred_nb <- predict(model_nb, newdata = titanictest)
#confusionMatrix(data=pred_nb, reference = titanic$Survived)
```

``` r
#library(ROCR)

#pred_test_nb <- predict(model_nb, newdata = titanictest, type="raw")
#p_test_nb <- prediction(pred_test_nb[,2], titanic$Survived)
#perf_nb <- performance(p_test_nb, "tpr", "fpr")
#plot(perf_nb, colorize=T)
#performance(p_test_nb, "auc")@y.values
```
