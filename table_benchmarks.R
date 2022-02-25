library("reshape2")
library("xtable")

all_res <- melt(lapply(0:4, function(i){
  read.csv(paste0("results/benchmarks/rep", i, ".csv"))
}))

avg <- aggregate(value ~ X + variable, data = all_res, 
                 FUN = mean)
avg$X[avg$X == "smm_ensemble"] <- "SMMwE"

std <- aggregate(value ~ X + variable, data = all_res, 
                 FUN = sd)
std$X[std$X == "smm_ensemble"] <- "SMMwE"



tab.avg <- acast(avg,  X ~ variable)
tab.std <- acast(std,  X ~ variable)
tab.all <- matrix(paste0(tab.avg, " (",format(tab.std, digits = 1),")"), 
                  nrow = nrow(tab.avg), 
                  dimnames = list(rownames(tab.avg), colnames(tab.avg)))
select <- c("SMMwE",  "rcc", "best")
idx <- which(rownames(tab.all) %in% select)
print(xtable(t(tab.all[idx,])))


