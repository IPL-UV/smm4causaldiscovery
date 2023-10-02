library(ggplot2)
source("plot_util.R")

## this should be colorblind-safe
colors <- palette.colors(7, palette = "R4")
names(colors) <- NULL

cols = c(
  "SMMwE" = colors[1],
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
noises <- c("normal2")
sizes <- c(250)
ntrains <- c(50, 100, 250, 500, 750, 1000)
ncoefs <- c(0.5)
ntests <- 1000

data <- load_results(mechs, noises,  ncoefs, sizes, ntrains, ntests, 
                     dir = "results", exp = "generated_data_mix", 5)

zalpha <- qnorm(0.975) # for 0.95 CI
D <- aggregate(value ~ mech + noise + ncoef + size + ntrain + ntest + alg + variable,
               data = data, FUN = function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)) )
D <- na.omit(do.call(data.frame, D))

D$alg[D$alg == "smm_ensemble"] <- "CEMM"
selected <- c("CEMM", "rcc", "jarfo", "best")
#selected <- c("smm_ensemble", "ANM", "CDS", "BivariateFit", "IGCI", "RECI")
plot_acc4_sigmoid <-
  ggplot(D[D$alg %in% selected & D$variable == "acc" & 
             D$mech %in% c("sigmoid_add", "sigmoid_mix"), ],
         aes(
           x = ntrain,
           y = value.mean,
           ymin = value.mean - zalpha*value.sd/sqrt(5),
           ymax = value.mean + zalpha*value.sd/sqrt(5),
           col = alg,
           fill = alg
         )) + geom_line() +
  facet_grid(cols = vars(mech), scales = "free_y") +
  geom_ribbon(alpha = 0.4, linetype = 0) +
  #ylim(0,1) + 
  scale_color_manual(values = cols[selected]) +
  scale_fill_manual(values = cols[selected]) +
  theme_bw() + labs(color = "method",
                    fill = "method",
                    y = "accuracy",
                    x = 'training size') +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 30))

fwidth <- 390 / 72
ggsave(paste0("images/accuracy_generated_4_sigmoid.pdf"),
       plot = plot_acc4_sigmoid, width = fwidth, height=2.5)



plot_acc4_gp <-
  ggplot(D[D$alg %in% selected & D$variable == "acc" & 
             D$mech %in% c("gp_add", "gp_mix"), ],
         aes(
           x = ntrain,
           y = value.mean,
           ymin = value.mean - zalpha*value.sd/sqrt(5),
           ymax = value.mean + zalpha*value.sd/sqrt(5),
           col = alg,
           fill = alg
         )) + geom_line() +
  facet_grid(cols = vars(mech), scales = "free_y") +
  geom_ribbon(alpha = 0.4, linetype = 0) +
  #ylim(0,1) + 
  scale_color_manual(values = cols[selected]) +
  scale_fill_manual(values = cols[selected]) +
  theme_bw() + labs(color = "method",
                    fill = "method",
                    y = "accuracy",
                    x = 'training size') +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 30))

ggsave(paste0("images/accuracy_generated_4_gp.pdf"),
       plot = plot_acc4_gp, width = 3.5, height=2.5)



DD <- aggregate(value ~ alg + ntrain + variable, data = data, FUN = mean)
DD$alg[DD$alg == "smm_ensemble"] <- "SMMwE"

plot_time <- ggplot(
  DD[DD$variable %in% c("t.train", "t.test"), ],
  aes(
    x = ntrain,
    y = value,
    group = alg,
    col = alg,
    fill = alg
  )
) + geom_line(size = 1) +
  facet_grid(cols = vars(variable), scales = "free_y") +
  #geom_ribbon(alpha = 0.4, linetype = 0) +
  scale_y_log10() +
  scale_color_manual(values = cols[selected]) +
  theme_bw() + labs(color = "method",
                    fill = "method",
                    y = "time (seconds)",
                    x = 'training size') +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 30))

ggsave(paste0("images/time_exp2.pdf"),
       plot = plot_time, width = 3.5, height = 2.5)
