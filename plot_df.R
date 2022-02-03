library(reshape2)

mechs <- c("nn")
sizes <- c(50, 100, 250, 500)
ntrains <- c(100)
ncoefs <- c(0.2, 0.4, 0.6)
ntests <- 100
gammas <- 1

D <- load_dfs(mechs, ncoefs, sizes, ntrains, ntests, gammas, 
             dir = "results", exp = "generated_data", 10)

ggplot(D) + geom_histogram(aes(x = value, color = variable, fill = variable),binwidth = 0.1) +
  facet_grid(rows = vars(variable), cols = vars(size)) + theme_bw() + ylab("") + 
  theme(axis.text.y = element_blank(), axis.ticks = element_blank()) + xlab("SMM decision function") 
