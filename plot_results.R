library(ggplot2)
library(colorblindr)
source("plot_util.R")

dir.create("images", showWarnings = FALSE)

mechs <- c("nn", "polynomial", "sigmoid_add", "sigmoid_mix")
sizes <- c(50, 100, 250, 500, 750, 1000)
ntrains <- c(100)
ncoefs <- c(0.2, 0.4, 0.6)
ntests <- 100
gammas <- 1

data <- load_results(mechs, ncoefs, sizes, ntrains, ntests, gammas, 
             dir = "results", exp = "generated_data", 10)

D <- aggregate(value ~ mech + ncoef + size + ntrain + ntest + gamma + alg + variable,
               data = data, FUN = mean )

## this should be colorblind-safe
colors <- palette.colors(7, palette = "R4")
names(colors) <- NULL

cols = c(
  "smm_ensemble" = colors[1],
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

plot_acc1 <-
  ggplot(D[D$alg %in% c("smm_ensemble", "avg", "vot", "best") & 
                D$variable == "acc", ],
         aes(
           x = size,
           y = value,
           col = alg,
           fill = alg
         )) + geom_line() +
  #geom_ribbon(alpha = 0.4, linetype = 0) +
  facet_grid(rows = vars(ncoef), cols = vars(mech), scales = "free_y") +
  ylim(0.5,1) + 
  scale_color_manual(values = cols) +
  scale_fill_manual(values = cols) +
  theme_bw() + labs(color = "algorithms",
                    fill = "algorithms",
                    y = "accuracy",
                    x = 'sample size') +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 30))

## check colorblinder
#colorblindr::view_cvd(plot_acc1)

# save plot
ggsave(
  filename = paste0("images/accuracy_generated_1.pdf"),
  plot = plot_acc1,
  width = 5,
  height = 4
)


plot_acc2 <- ggplot(
  D[D$alg %in% c("smm_ensemble", "rcc", "jarfo")  & 
    D$variable == "acc",],
  aes(
    x = size,
    y = value,
    group = alg,
    col = alg,
    fill = alg
  )
) + geom_line() +
  ylim(0.5,1) + 
  #geom_ribbon(alpha = 0.4, linetype = 0) +
  facet_grid(rows = vars(ncoef), cols = vars(mech), scales = "free_y") +
  scale_color_manual(values = cols) +
  scale_fill_manual(values = cols) +
  theme_bw() + labs(color = "algorithms",
                    fill = "algorithms",
                    y = "accuracy",
                    x = 'sample size') +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 30))

ggsave(
  filename = paste0("images/accuracy_generated_2.pdf"),
  plot = plot_acc2,
  width = 5,
  height = 4
)

plot_acc3 <- ggplot(
  D[D$alg %in% c("smm_ensemble", "ANM", "CDS", "BivariateFit", "IGCI", "RECI") & 
      D$variable == "acc", ],
  aes(
    x = size,
    y = value,
    group = alg,
    col = alg,
    fill = alg
  )
) + geom_line() +
  ylim(0.5,1) + 
  scale_color_manual(values = cols) +
  scale_fill_manual(values = cols) +
  #geom_ribbon(alpha = 0.4, linetype = 0) +
  facet_grid(rows = vars(ncoef), cols = vars(mech), scales = "free_y") +
  theme_bw() + labs(color = "algorithms",
                    fill = "algorithms",
                    y = "accuracy",
                    x = 'sample size') +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 30))

# save plot
ggsave(
  filename = paste0("images/accuracy_generated_3.pdf"),
  plot = plot_acc3,
  width = 5,
  height = 4
)


DD <- aggregate(value ~ alg + size + variable, data = data, FUN = mean)

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
  theme_bw() + labs(color = "algorithms",
                                                    fill = "algorithms",
                                                    y = "time (seconds)",
                                                    x = 'training size') +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 30))

ggsave(paste0("images/time_exp.pdf"),
       plot = plot_time)



mechs <- c("nn")
sizes <- c(250)
ntrains <- c(100, 250, 500, 750, 1000)
ncoefs <- c(0.2, 0.4)
ntests <- 1000
gammas <- 1

data <- load_results(mechs, ncoefs, sizes, ntrains, ntests, gammas, 
                     dir = "results", exp = "generated_data", 10)

D <- aggregate(value ~ mech + ncoef + size + ntrain + ntest + gamma + alg + variable,
               data = data, FUN = function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)) )
D <- do.call(data.frame, D)

plot_acc4 <-
  ggplot(D[D$alg %in% c("smm_ensemble", "rcc", "jarfo", "best") & 
             D$variable == "acc", ],
         aes(
           x = ntrain,
           y = value.mean,
           ymin = value.mean - value.sd,
           ymax = value.mean + value.sd,
           col = alg,
           fill = alg
         )) + geom_line() +
  facet_grid(rows = vars(ncoef), cols = vars(mech), scales = "free_y") +
  geom_ribbon(alpha = 0.4, linetype = 0) +
  #ylim(0,1) + 
  scale_color_manual(values = cols) +
  scale_fill_manual(values = cols) +
  theme_bw() + labs(color = "algorithms",
                    fill = "algorithms",
                    y = "accuracy",
                    x = 'training size') +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 30))

ggsave(paste0("images/accuracy_generated_4.pdf"),
       plot = plot_acc4)
