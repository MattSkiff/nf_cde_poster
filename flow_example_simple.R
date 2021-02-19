# Simple Flow visualisation

library(ggplot2)
library(MASS)
library(gridExtra)
library(lattice)
library(grid)

n <- 10000

# sampling z from base distribution
mvn.mat <- mvrnorm(n = n,mu = c(0,0),Sigma = diag(2),tol = 1e-6,empirical = TRUE)

mvr_sample.df <- as.data.frame(mvn.mat)

p1 <- ggplot(mvr_sample.df) +
        geom_point(mapping = aes(x = V1,y = V2),alpha = 0.1) +
        theme_light() +
        labs(x = "x",y = "y",title = "Sample from MVN",caption = "n = 10k",subtitle = "Base distribution") +
        scale_x_continuous(limits = c(-5,5), expand = c(0, 0)) +
        scale_y_continuous(limits = c(-5,5), expand = c(0, 0))  + 
        theme(legend.position = "none") 

# diagonal matrix for A
A_diag <- diag(x = runif(n,min = 0,max = 3))
b_diag <- as.integer(runif(1,min = -3, max = 3))

# triangular matrix for A - more expressive
A_tri <- matrix(0, n,n)
# source: https://stackoverflow.com/questions/9282258/how-to-fill-matrix-with-random-numbers-in-r/9282447
A_tri[row(A_tri)+col(A_tri) >=n+1] <- (row(A_tri)+col(A_tri) -n+1)[row(A_tri)+col(A_tri)>=n+1]/n
A_tri

# random triangular matrix - exp dist
A_tri_exp <- upper.tri(matrix(rexp(n*n, rate=.1), ncol=n))

linear_transform <- function(x,A,b) {
  return(A %*% x + b)
}

# generative direction
mvn_linear_transform_A_diag.mat <- linear_transform(x = mvn.mat,A = A_diag,b = b_diag)
mvn_linear_transform_A_diag.df <- as.data.frame(mvn_linear_transform_A_diag.mat)

mvn_linear_transform_A_tri_exp.mat <- linear_transform(x = mvn.mat,A = A_tri_exp,b = b_diag)
mvn_linear_transform_A_tri_exp.df <- as.data.frame(mvn_linear_transform_A_tri_exp.mat)

p2 <- ggplot(mvn_linear_transform_A_diag.df) +
        geom_point(mapping = aes(x = V1,y = V2),alpha = 0.1) +
        theme_light() +
        labs(x = "x",y = "y",title = "Linear transform on sample",caption = "n = 10k",subtitle = "Using diagonal matrix for transformation matrix (A)") +
        scale_x_continuous(expand = c(0, 0)) +
        scale_y_continuous(expand = c(0, 0))  + 
        theme(legend.position = "none")

p3 <- ggplot(mvn_linear_transform_A_tri_exp.df) +
  geom_point(mapping = aes(x = V1,y = V2),alpha = 0.1) +
  theme_light() +
  labs(x = "x",y = "y",title = "Linear transform on sample",caption = "n = 10k",subtitle = "Using triangular matrix for A (sampled from exp. dist.)") +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0))  + 
  theme(legend.position = "none")

plot_explanation <- "Example of a simple normalising flow\n in the generative direction for linear flows,\n illustrating the difference in expressiveness\n between diagonal and triangular matrices"
t <- textGrob(plot_explanation)
final <- grid.arrange(p1,p2,p3,t,nrow = 2)
ggsave("linear_flow_diag.png",plot = final,width = 25,height = 25,units = "cm")



