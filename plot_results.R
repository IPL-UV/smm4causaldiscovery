library(ggplot2)
dir.create("images", showWarnings = FALSE)

mechs <- c("nn", "gp_mix" , "gp_add", "polynomial",  "sigmoid_mix", "sigmoid_add")
sizes <- c(50, 100, 200, 300 ,400)
n <- 100
ncoefs <- c(0.2, 0.4, 0.6)


Dacc <- Dtrain <- Dtest <- 
  data.frame(ALG = double(0), MEAN = double(0), SEM = double(0), SIZE = double(0), MECH = double(0))

for (ncoef in ncoefs){
  for (mech in mechs){
    for (s in sizes){
      basepath <- paste0("experiments/", mech, "_n", n, "_s", s, "_ncoeff", ncoef)
      files <- file.path(basepath, paste0("run", 1:5, ".csv"))
      res <- lapply(files, function(x) c(read.csv(x)))
      accs <- t(sapply(res, function(x) sapply(x, function(xx) xx[1])))
      train <- t(sapply(res, function(x) sapply(x, function(xx) xx[2])))
      test <- t(sapply(res, function(x) sapply(x, function(xx) xx[3])))
      Dacc <- rbind(Dacc, data.frame(ALG = colnames(accs), MEAN = colMeans(accs), 
                                     SEM = apply(accs, 2, sd) / sqrt(nrow(accs)),
                                     SIZE = s, MECH = mech, NCOEF = ncoef))
      Dtrain <- rbind(Dtrain, data.frame(ALG = colnames(train), MEAN = colMeans(train, na.rm = TRUE), 
                                         SEM = apply(train, 2, sd) / sqrt(nrow(train)),
                                         SIZE = s, MECH = mech, NCOEF = ncoef))
      Dtest <- rbind(Dtest, data.frame(ALG = colnames(test), MEAN = colMeans(test, na.rm = TRUE), 
                                       SEM = apply(test, 2, sd) / sqrt(nrow(test)),
                                       SIZE = s, MECH = mech, NCOEF = ncoef))
    }
  }  
}




ggplot(Dacc, aes(x = SIZE, y = MEAN, ymax = MEAN + SEM, ymin = MEAN - SEM, 
              group = ALG, col = ALG, fill = ALG)) + geom_line() + 
              geom_ribbon(alpha = 0.4, linetype = 0) + 
               facet_grid(cols = vars(MECH), rows = vars(NCOEF)) + 
  theme_bw() + labs(color = "algorithms", fill = "algorithms", y = "accuracy", 
                                                                 x = 'sample size') + 
              theme(legend.position = "bottom", axis.text.x = element_text(angle = 30)) +
  ggsave(paste0("images/n", n,  "_accuracy_exp.pdf"), width = 6, height = 4)



ggplot(Dtrain, aes(x = SIZE, y = MEAN, ymax = MEAN + SEM, ymin = MEAN - SEM, 
                 group = ALG, col = ALG, fill = ALG)) + geom_line() + 
  #geom_ribbon(alpha = 0.4, linetype = 0) + 
  scale_y_log10() + 
  facet_grid(cols = vars(MECH), rows = vars(NCOEF)) + theme_bw() + labs(color = "algorithms", fill = "algorithms", y = "training time", 
                                                    x = 'sample size') + 
  ggsave(paste0("images/n", n,  "_training_time_exp.pdf"))


ggplot(Dtest, aes(x = SIZE, y = MEAN, ymax = MEAN + SEM, ymin = MEAN - SEM, 
                   group = ALG, col = ALG, fill = ALG)) + geom_line() + 
  #geom_ribbon(alpha = 0.4, linetype = 0) + 
  scale_y_log10() + 
  facet_grid(cols = vars(MECH), rows = vars(NCOEF)) + theme_bw() + labs(color = "algorithms", fill = "algorithms", y = "testing time", 
                                                    x = 'sample size') + 
  ggsave(paste0("images/n", n,  "_testing_time_exp.pdf"))
