# Example visualisation of kernel CDE on old faithful data

library(hdrcde)
library(ggplot2)
library(gridExtra)

faithful.cde <- cde(x = faithful$waiting, y = faithful$eruptions,
                    x.name="Waiting time", y.name = "Duration time",
                    x.margin = 80)

faithful2.cde <- cde(x = faithful$waiting, y = faithful$eruptions,
                     x.name="Waiting time", y.name = "Duration time",
                     x.margin = 50)

cde_df <- data.frame(y = faithful.cde$y,
                     z = as.vector(faithful.cde$z),
                     z2 = as.vector(faithful2.cde$z))

p1 <- ggplot(data = cde_df) +
  geom_line(mapping = aes(x = y,y = z),color = 'red') +
  geom_line(mapping = aes(x = y,y = z2),color = 'blue') +
  theme_light() +
  labs(title = "Conditional Density Estimates (2)\n of Eruption Duration",
       x = "target",y = "density")

p2 <- ggplot(data = faithful) +
  geom_point(mapping = aes(x = waiting,y = eruptions)) +
  geom_vline(xintercept = 80,color = 'red') + 
  geom_vline(xintercept = 50,color = 'blue') + 
  theme_light() +
  labs(title = "Scatterplot of Old Faithful Data\n and CDE query points")

grid.arrange(p1, p2, nrow = 1)