library(ggplot2)
library(colorblindr)
source("plot_util.R")

## this should be colorblind-safe
colors <- palette.colors(7, palette = "R4")
names(colors) <- NULL

cols = c(
  "CEMM" = colors[1],
  "avg" = colors[2],
  "vot" = colors[3],
  "best" = colors[4],
  ####
  "rcc" = colors[2],
  "jarfo" = colors[3],
  ####
  "ANM" = colors[2],
  "CDS" = colors[3],
  "BivariateFit" = colors[4],
  "IGCI" = colors[5],
  "RECI" = colors[6]
)

dir.create("images", showWarnings = FALSE)

mechs <- c("nn", "polynomial", "sigmoid_add", "sigmoid_mix", "gp_add", "gp_mix")
noises <- c("normal2", "uniform2")
sizes <- c(50, 100, 250, 500, 750, 1000)
ntrains <- c(100)
ncoefs <- c("0.5")
ntests <- 100

data <- load_results(mechs, noises, ncoefs, sizes, ntrains, ntests,  
             dir = "results", exp = "generated_data", 10)

D <- aggregate(value ~ mech + ncoef + size + ntrain + ntest + alg + variable + noise,
               data = data, FUN = mean )

D$noise[D$noise == "normal2"] <- "normal" 
D$noise[D$noise == "uniform2"] <- "uniform" 
D$alg[D$alg=="smm_ensemble"] <- "CEMM"
selected <- c("CEMM", "avg", "vot", "best")
plot_acc1 <-
  ggplot(D[D$alg %in% selected & 
                D$variable == "acc", ],
         aes(
           x = size,
           y = value,
           col = alg,
           fill = alg
         ))  + geom_line() + 
  #geom_ribbon(alpha = 0.4, linetype = 0) +
  facet_grid(rows = vars(noise), cols = vars(mech), scales = "free_y") +
  #ylim(0.5,1) + 
  scale_color_manual(values = cols) +
  scale_fill_manual(values = cols) +
  theme_bw() + labs(color = "method",
                    fill = "method",
                    y = "accuracy",
                    x = 'sample size') +
  coord_cartesian(ylim = c(0.4,1)) + 
  theme(legend.position = "bottom",
	legend.title=element_blank(),
        axis.text.x = element_text(angle = 30))

## check colorblinder
#colorblindr::view_cvd(plot_acc1)

# save plot
ggsave(
  filename = paste0("images/accuracy_generated_1.pdf"),
  plot = plot_acc1,
  width = 6,
  height = 4
)



selected <- c("CEMM", "rcc", "jarfo", "best") 
plot_acc2 <- ggplot(
  D[D$alg %in% selected & 
    D$variable == "acc",],
  aes(
    x = size,
    y = value,
    group = alg,
    col = alg,
    fill = alg
  )
) + geom_line() +
  #ylim(0.5,1) + 
  #geom_ribbon(alpha = 0.4, linetype = 0) +
  facet_grid(rows = vars(noise), cols = vars(mech), scales = "free_y") +
  scale_color_manual(values = cols[selected]) +
  scale_fill_manual(values = cols[selected]) +
  theme_bw() + labs(color = "method",
                    fill = "method",
                    y = "accuracy",
                    x = 'sample size') +
  coord_cartesian(ylim = c(0.25,1)) + 
  theme(legend.position = "bottom",
	legend.title=element_blank(),
        axis.text.x = element_text(angle = 30))

ggsave(
  filename = paste0("images/accuracy_generated_2.pdf"),
  plot = plot_acc2,
  width = 6,
  height = 4
)

plot_acc2_mini <- ggplot(
  D[D$alg %in% selected & 
    D$mech %in% c('nn', 'polynomial', 'sigmoid_mix') & 
    D$variable == "acc",],
  aes(
    x = size,
    y = value,
    group = alg,
    col = alg,
    fill = alg
  )
) + geom_line() +
  #ylim(0.5,1) + 
  #geom_ribbon(alpha = 0.4, linetype = 0) +
  facet_grid(rows = vars(noise), cols = vars(mech), scales = "free_y") +
  scale_color_manual(values = cols[selected]) +
  scale_fill_manual(values = cols[selected]) +
  theme_bw() + labs(color = "method",
                    fill = "method",
                    y = "accuracy",
                    x = 'sample size') +
  coord_cartesian(ylim = c(0.25,1)) + 
  theme(legend.position = "bottom",
	legend.title=element_blank(),
        axis.text.x = element_text(angle = 30))

ggsave(
  filename = paste0("images/accuracy_generated_2_mini.pdf"),
  plot = plot_acc2_mini,
  width = 3.5,
  height = 3.5
)

selected <- c("CEMM", "ANM", "CDS", "BivariateFit", "IGCI", "RECI") 
plot_acc3 <- ggplot(
  D[D$alg %in% selected &  
      D$variable == "acc", ],
  aes(
    x = size,
    y = value,
    group = alg,
    col = alg,
    fill = alg
  )
) + geom_line() +
  #ylim(0.5,1) + 
  scale_color_manual(values = cols[selected]) +
  scale_fill_manual(values = cols[selected]) +
  #geom_ribbon(alpha = 0.4, linetype = 0) +
  facet_grid(rows = vars(noise), cols = vars(mech), scales = "free_y") +
  theme_bw() + labs(color = "method",
                    fill = "method",
                    y = "accuracy",
                    x = 'sample size') +
  coord_cartesian(ylim = c(0.25,1)) + 
  theme(legend.position = "bottom",
	legend.title=element_blank(),
        axis.text.x = element_text(angle = 30))

# save plot
ggsave(
  filename = paste0("images/accuracy_generated_3.pdf"),
  plot = plot_acc3,
  width = 6,
  height = 4
)


DD <- aggregate(value ~ alg + size + variable, data = data, FUN = mean)

DD$alg[DD$alg=="smm_ensemble"] <- "CEMM"
plot_time <- ggplot(
  DD[DD$variable %in% c("t.train", "t.test"), ],
  aes(
    x = size,
    y = value,
    group = alg,
    col = alg,
    fill = alg
  )
) + geom_line(size = 1) +
  facet_grid(cols = vars(variable), scales = "free_y") +
  #geom_ribbon(alpha = 0.4, linetype = 0) +
  scale_y_log10() +
  scale_color_manual(values = cols) +
  theme_bw() + labs(color = "method",
		    fill = "method",
		    y = "time (seconds)",
		    x = 'sample size') +
  theme(legend.position = "bottom",
	legend.title=element_blank(),
	axis.text.x = element_text(angle = 30))

ggsave(paste0("images/time_exp.pdf"),
       plot = plot_time, width = 2.5, height = 2.5)
