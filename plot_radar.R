library("ggplot2")
source("plot_util.R")

mechs <- c("nn", "polynomial", "sigmoid_add", "sigmoid_mix", "gp_add", "gp_mix")
sizes <- c(50, 100, 250, 500)
ntrains <- c(100)
ncoefs <- c(0.2, 0.4, 0.6)
ntests <- 100
gammas <- 1

data <- load_results(mechs, ncoefs, sizes, ntrains, ntests, gammas, 
                     dir = "results", exp = "generated_data", 10)
D <- aggregate(value ~ mech + size + alg + variable,
               data = data, FUN = mean )

ggplot(D[D$size == 500 & D$variable == "acc" & D$alg %in% c("smm_ensemble", "best", "avg", "vot"),], 
       aes(x = mech, y = value, colour = alg, group = alg, fill = alg)) +
  geom_polygon(alpha = 0.2) +
  #facet_grid( rows = vars(size)) + 
  coord_polar(theta = "x", direction = -1, clip = "off")  +
  theme_bw() + ylim(c(0, 1)) + 
  theme(panel.border = element_blank(),
        axis.text.y = element_blank(), 
        axis.ticks = element_blank(), axis.title = element_blank(),
        legend.position = "bottom")
