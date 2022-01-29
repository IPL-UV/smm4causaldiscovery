library(ggplot2)
dir.create("images", showWarnings = FALSE)

mechs <- c("nn")
sizes <- c(50, 100, 250, 500, 1000)
ns <- c(100)
ncoefs <- c(0.4)


Dacc <- Dtrain <- Dtest <- 
  data.frame(ALG = double(0), MEAN = double(0), SEM = double(0), SIZE = double(0), MECH = double(0))

for (n in ns){
  for (ncoef in ncoefs){
    for (mech in mechs){
      for (s in sizes){
        basepath <- file.path("results", "generated_data", 
			      paste0(mech,"_s", s, "_ntrain", n))
        files <- file.path(basepath, paste0("rep", 0:9, ".csv"))
        res <- lapply(files, function(x) c(read.csv(x)))
        accs <- t(sapply(res, function(x) sapply(x, function(xx) xx[1])))
        #train <- t(sapply(res, function(x) sapply(x, function(xx) xx[2])))
        #test <- t(sapply(res, function(x) sapply(x, function(xx) xx[3])))
        Dacc <- rbind(Dacc, data.frame(ALG = colnames(accs), MEAN = colMeans(accs), 
                                       SEM = apply(accs, 2, sd) / sqrt(nrow(accs)),
				       N = n, 
                                       SIZE = s, MECH = mech, NCOEF = ncoef))
        #Dtrain <- rbind(Dtrain, data.frame(ALG = colnames(train), MEAN = colMeans(train, na.rm = TRUE), 
#                                           SEM = apply(train, 2, sd) / sqrt(nrow(train)),
#					   N = n,
#                                           SIZE = s, MECH = mech, NCOEF = ncoef))
        #Dtest <- rbind(Dtest, data.frame(ALG = colnames(test), MEAN = colMeans(test, na.rm = TRUE), 
#                                         SEM = apply(test, 2, sd) / sqrt(nrow(test)),
#					 N = n,
#                                         SIZE = s, MECH = mech, NCOEF = ncoef))
      }
    }  
  }
}



ggplot(Dacc, aes(x = SIZE, y = MEAN, ymax = MEAN + SEM, ymin = MEAN - SEM, 
              group = ALG, col = ALG, fill = ALG)) + geom_line() + 
              geom_ribbon(alpha = 0.4, linetype = 0) + 
               facet_grid(cols = vars(MECH), scales = "free_y") + 
  theme_bw() + labs(color = "algorithms", fill = "algorithms", y = "accuracy", 
                                                                 x = 'sample size') + 
              theme(legend.position = "bottom", axis.text.x = element_text(angle = 30)) +
  ggsave(paste0("images/accuracy_exp.pdf"), width = 6, height = 4)



#ggplot(Dtrain, aes(x = N, y = MEAN, ymax = MEAN + SEM, ymin = MEAN - SEM, 
#                 group = ALG, col = ALG, fill = ALG)) + geom_line() + 
#  #geom_ribbon(alpha = 0.4, linetype = 0) + 
#  scale_y_log10() + 
#  facet_grid(cols = vars(MECH), rows = vars(SIZE)) + theme_bw() + labs(color = "algorithms", fill = "algorithms", y = "training time", 
#                                                    x = 'training size') + 
#  ggsave(paste0("images/training_time_exp.pdf"))


#ggplot(Dtest, aes(x = N, y = MEAN, ymax = MEAN + SEM, ymin = MEAN - SEM, 
#                   group = ALG, col = ALG, fill = ALG)) + geom_line() + 
#  #geom_ribbon(alpha = 0.4, linetype = 0) + 
#  scale_y_log10() + 
#  facet_grid(cols = vars(MECH), rows = vars(SIZE)) + theme_bw() + labs(color = "algorithms", fill = "algorithms", y = "testing time", 
#                                                    x = 'training size') + 
#  ggsave(paste0("images/testing_time_exp.pdf"))
