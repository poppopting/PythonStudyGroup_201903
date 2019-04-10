#1
f <- function(x){
  x^2 - 3
}
delta <- function(x){
  -(x^2-3) / (2*x)
}
x <- c()
x[1] <- 2
for(i in 2:100){
  x[i] <- x[i-1] + delta(x[i-1])
  if(abs(f(x[i]) < 10^-6)) break
}
x

#2
f1 <- function(x){
  cos(x) - x
}
delta1 <- function(x){
  -(cos(x)-x) / (-sin(x)-1)
}

x1 <- c()
x1[1] <- pi/4
for(i in 2:100){
  x1[i] <- x1[i-1] + delta1(x1[i-1])
  if(abs(f1(x[i]) < 10^-6)) break
}
x1
x1[length(x1)]

#3
f2 <- function(x){
  x^2 - 2*x + 1
}
delta2 <- function(x){
  -(x^2 - 2*x + 1) / (2*x - 2)
}

x2 <- c()
x2[1] <- 2
for(i in 2:100){
  x2[i] <- x2[i-1] + delta2(x2[i-1])
  if(abs(f2(x2[i]) < 10^-7)) break
}
x2
x2[length(x2)]

#1
#gradient descent
f3 <- function(x){
  x^3 - 2*x^2 + 2
}
delta3 <- function(x){
  3*x^2 - 4*x
}

r <- 0.1
x <- c()
x[1] <- 2
for(i in 1:100){
  x[i+1] <- x[i] - r*delta3(x[i])
  if(abs(x[i+1] - x[i]) < 10^-7) break
}
x
x[length(x)]

y <- c()
y[1] <- 0.01
for(i in 1:100){
  y[i+1] <- y[i] - r*delta3(y[i])
  if(abs(y[i+1] - y[i]) < 10^-7) break
}
y
y[length(y)] # iterate more times

#newton
delta31 <- function(x){
  -(3*x^2 - 4*x) / (6*x - 4)
}
x11 <- c()
x11[1] <- 2
for(i in 2:100){
  x11[i] <- x11[i-1] + delta31(x11[i-1])
  if(abs(f3(x11[i]) < 10^-6)) break
}
x11
x11[length(x11)]

y11 <- c()
y11[1] <- 0.01
for(i in 2:100){
  y11[i] <- y11[i-1] + delta31(y11[i-1])
  if(abs(f3(y11[i]) < 10^-6)) break
}
y11
y11[length(y11)] # find local maximum

