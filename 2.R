library('class')        # функция knn()
library('e1071')        # функция naiveBayes()
library('MASS')         # функция mvrnorm()

# ядро
my.seed <- 12345

n <- 100               # наблюдений всего
train.percent <- 0.85  # доля обучающей выборки

# x-ы -- двумерные нормальные случайные величины
set.seed(my.seed)
class.0 <- mvrnorm(45, mu = c(5, 12), 
                   Sigma = matrix(c(6^2, 0, 0, 13.7^2), 2, 2, 
                                  byrow = T))

set.seed(my.seed + 1)
class.1 <- mvrnorm(55, mu = c(7, 22), 
                   Sigma = matrix(c(4.3^2, 0, 0, 21^2), 2, 2, 
                                  byrow = T))

# записываем x-ы в единые векторы (объединяем классы 0 и 1)
x1 <- c(class.0[, 1], class.1[, 1])
x2 <- c(class.0[, 2], class.1[, 2])

# фактические классы Y
y <- c(rep(0, nrow(class.0)), rep(1, nrow(class.1)))

# классы для наблюдений сетки
rules <- function(x1, x2){
  ifelse(x2 < 1.6*x1 + 19, 0, 1)
}
# Конец данных примера 3 .......................................................


# Отбираем наблюдения в обучающую выборку --------------------------------------
set.seed(my.seed)
inTrain <- sample(seq_along(x1), train.percent*n)
x1.train <- x1[inTrain]
x2.train <- x2[inTrain]
x1.test <- x1[-inTrain]
x2.test <- x2[-inTrain]

# используем истинные правила, чтобы присвоить фактические классы
y.train <- y[inTrain]
y.test <- y[-inTrain]

# фрейм с обучающей выборкой
df.train.1 <- data.frame(x1 = x1.train, x2 = x2.train, y = y.train)
# фрейм с тестовой выборкой
df.test.1 <- data.frame(x1 = x1.test, x2 = x2.test
                        
                        # Рисуем обучающую выборку графике ---------------------------------------------
                        
                        # для сетки (истинных областей классов): целочисленные значения x1, x2
             
                        # цвета для графиков
  cls <- c('blue', 'orange')
   cls.t <- c(rgb(0, 0, 1, alpha = 0.5), rgb(1,0.5,0, alpha = 0.5))
                        
                        # график истинных классов
        plot(df.train.1$x1, df.train.1$x2,
             pch = 21, col = cls.t[df.train.1[, 'y'] + 1],
bg = cls.t[df.train.1[, 'y'] + 1],
xlab = 'X1', ylab = 'Y1',
main = 'Обучающая выборка, факт' )
                  
                        # Байесовский классификатор ----------------------------------------------------
                        #  наивный байес: непрерывные объясняющие переменные
                   
                        # строим модель
                        nb <- naiveBayes(y ~ ., data = df.train.1)
                        # получаем модельные значения на обучающей выборке как классы
                        y.nb.train <- ifelse(predict(nb, df.train.1[, -3], 
                                                     type = "raw")[, 2] > 0.5, 1, 0)
                        
                        # график истинных классов
                        plot(df.train.1$x1, df.train.1$x2, 
                               pch = 21, bg = cls.t[y.nb.train + 1], 
                               col = cls.t[y.nb.train + 1],  xlab = 'X1', ylab = 'Y1',
                             main = 'Обучающая выборка, модель naiveBayes')                        
                        # матрица неточностей на обучающей выборке
                        tbl <- table(y.train, y.nb.train)
                        tbl                        
                        # точность, или верность (Accuracy)
                        Acc <- sum(diag(tbl)) / sum(tbl)
                        Acc                        
                        # прогноз на тестовую выборку
                        y.nb.test <- ifelse(predict(nb, df.test.1, type = "raw")[, 2] > 0.5, 1, 0)
                        
                        # матрица неточностей на тестовой выборке
                        tbl <- table(y.test, y.nb.test)
                        tbl                        
                        # точность, или верность (Accuracy)
                        Acc <- sum(diag(tbl)) / sum(tbl)
                        Acc                        
                        # Метод kNN --------------------------------------------------------------------
                        #  k = 3
                        
                        # строим модель и делаем прогноз
                        y.knn.train <- knn(train = scale(df.train.1[, -3]), 
                                           test = scale(df.train.1[, -3]),
                                           cl = df.train.1$y, k = 3)
                        
                        # график истинных классов
                        plot(df.grid.1$x1, df.grid.1$x2, 
                             pch = '·', col = cls[df.grid.1[, 'y'] + 1],
                             xlab = 'X1', ylab = 'Y1',
                             main = 'Обучающая выборка, модель kNN')
                        # точки наблюдений, предсказанных по модели
                        points(df.train.1$x1, df.train.1$x2, 
                               pch = 21, bg = cls.t[as.numeric(y.knn.train)], 
                               col = cls.t[as.numeric(y.knn.train)])                        
                        # матрица неточностей на обучающей выборке
                        tbl <- table(y.train, y.knn.train)
                        tbl                        
                        # точность (Accuracy)
                        Acc <- sum(diag(tbl)) / sum(tbl)
                        Acc                        
                        # прогноз на тестовую выборку
                        y.knn.test <- knn(train = scale(df.train.1[, -3]), 
                                          test = scale(df.test.1[, -3]),
                                          cl = df.train.1$y, k = 3)
                        
                        # матрица неточностей на тестовой выборке
                        tbl <- table(y.test, y.knn.test)
                        tbl                        
                        # точность (Accuracy)
                        Acc <- sum(diag(tbl)) / sum(tbl)
                        Acc                        
                        
                        
                        
                        
                        
                
                        