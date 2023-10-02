library("reshape2")
library("ggplot2")
library("ggridges")
source("plot_util.R")

mechs <- c("linear", "nn", "polynomial", "sigmoid_add", 
           "sigmoid_mix", "gp_add", "gp_mix")
sizes <- c(250)
ntrains <- c(500, 750, 1000)
noises <- c("normal2")
ncoefs <- c("0.5")
ntests <- 1000
gammas <- NA

D <- load_dfs(mechs, noises, ncoefs, sizes, ntrains, ntests,
             dir = "results", exp = "generated_data_mix", 10)

D <- na.omit(D[!D$variable == "X_constant", ])

colors <- palette.colors(7, palette = "R4")
names(colors) <- NULL

cols = c(
  "ANM" = colors[2],
  "CDS" = colors[3],
  "BivariateFit" = colors[4],
  "IGCI" = colors[5],
  "RECI" = colors[6]
)
pp <- ggplot(D, aes(x = value, y =variable,  fill = variable)) + 
	stat_density_ridges(bandwidth = 0.05, alpha = 0.8) + 
	facet_grid(cols = vars(mech)) + 
	coord_cartesian(clip = "off") +
	scale_y_discrete(expand = c(0.1, 0)) + 
	scale_fill_manual(values = cols) + 
	xlim(-2.2,2.2) + 
	#scale_x_continuous(expand = c(0,0)) +
	xlab("CMM decision function") + ylab("") + 
	labs(fill = "method") + 
	#theme_ridges() + 
	theme_bw() + 
	theme(legend.position = "none")


fwidth <- 390 / 72
ggsave(pp, file = "images/plot_df.pdf", width = fwidth, height = 3)
