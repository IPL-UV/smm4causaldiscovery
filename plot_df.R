library("reshape2")
library("ggplot2")
source("plot_util.R")

mechs <- c("nn", "polynomial", "sigmoid_add", "sigmoid_mix", "gp_add", "gp_mix")
sizes <- c(250)
ntrains <- c(100, 500, 1000)
ncoefs <- c(0.4)
ntests <- 1000
gammas <- 1

D <- load_dfs(mechs, ncoefs, sizes, ntrains, ntests, gammas, 
             dir = "results", exp = "generated_data", 10)

pp <- ggplot(D) + geom_density(aes(x = value, color = as.factor(ntrain), group = as.factor(ntrain)), bw = 0.05) +
  facet_grid(rows = vars(variable), cols = vars(mech)) + theme_bw() + ylab("") + 
  coord_cartesian(ylim=c(0, 2), xlim=c(-2,2)) #+
  #theme(axis.text.y = element_blank(), axis.ticks = element_blank()) + xlab("SMM decision function") 

 ggsave(pp, file = "images/plot_df.pdf")
